"""
Sobol Global Sensitivity Analysis
==================================
Variance-decomposition sensitivity on six parameters of the Fc-fusion model.
Outputs first-order (S1) and total-order (ST) indices for two PD readouts:
AUC of target occupancy, and days with TO > 50%.

Result: R0 (target abundance) dominates with ST ~0.51; k_ao (aqueous outflow)
has ST ~0 - i.e., it doesn't matter for the model output once it's "fast enough"
relative to vitreous-aqueous diffusion. This negative result tells the program
not to invest in precise k_ao characterization.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from model.tmdd_model import (
    params_for_format, simulate, dose_nmol_from_mg, TARGET_BIO,
)

# ==========================================================================
# Problem definition
# ==========================================================================
problem = {
    'num_vars': 6,
    'names': ['k_va', 'k_ao', 'k_vr', 'f_rec', 'k_on', 'R0_nM'],
    'bounds': [
        [np.log(0.0030), np.log(0.030)],   # k_va: log-uniform
        [np.log(0.10),   np.log(2.0)],     # k_ao
        [np.log(0.0003), np.log(0.005)],   # k_vr
        [0.05, 0.95],                       # f_rec: linear
        [np.log(0.01),   np.log(1.0)],     # k_on
        [np.log(0.5),    np.log(50)],      # R0
    ],
}
LOG_INDICES = [0, 1, 2, 4, 5]  # f_rec at index 3 stays linear

# ==========================================================================
# Run Sobol
# ==========================================================================
N = 64  # gives 64 * (2*6+2) = 896 evaluations
samples = sobol_sample.sample(problem, N, calc_second_order=True)
n_runs = samples.shape[0]
print(f"Running {n_runs} model evaluations for Sobol analysis...")

base_p = params_for_format('fc', species='rabbit')
base_dose = dose_nmol_from_mg(2.0, 'fc')

Y_AUC_TO = np.zeros(n_runs)
Y_days_50 = np.zeros(n_runs)

for i in range(n_runs):
    p = deepcopy(base_p)
    R0_val = None
    for j, name in enumerate(problem['names']):
        v = samples[i, j]
        if j in LOG_INDICES:
            v = np.exp(v)
        if name == 'R0_nM':
            R0_val = v
            p['k_syn'] = p['k_deg'] * v
        else:
            p[name] = v

    res = simulate(p, base_dose, t_max_d=45, n_pts=300, R0_override=R0_val)
    if res is None:
        continue

    Y_AUC_TO[i] = float(np.trapezoid(res['TO'], res['t_d']))

    above = (res['TO'] > 0.5).astype(int)
    if above.any():
        transitions = np.diff(np.concatenate([[0], above, [0]]))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        Y_days_50[i] = sum(res['t_d'][e-1] - res['t_d'][s]
                           for s, e in zip(starts, ends))

    if (i+1) % 200 == 0:
        print(f"  {i+1}/{n_runs}")

# ==========================================================================
# Analyze and save
# ==========================================================================
Si_AUC = sobol_analyze.analyze(problem, Y_AUC_TO, calc_second_order=True,
                                print_to_console=False)
Si_days = sobol_analyze.analyze(problem, Y_days_50, calc_second_order=True,
                                 print_to_console=False)

df = pd.DataFrame({
    'parameter': problem['names'],
    'S1_AUC_TO': Si_AUC['S1'],     'ST_AUC_TO': Si_AUC['ST'],
    'S1_days_50': Si_days['S1'],   'ST_days_50': Si_days['ST'],
})
df.to_csv('results/tables/sobol_indices.csv', index=False)
print("\nSobol indices:")
print(df.to_string(index=False))

# ==========================================================================
# Plot
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(len(problem['names'])); w = 0.35

for ax, (Si, title) in zip(axes,
                            [(Si_AUC, 'AUC of target occupancy'),
                             (Si_days, 'Days with TO > 50%')]):
    ax.bar(x - w/2, Si['S1'], w, label='First-order $S_1$', color='#1D9E75')
    ax.bar(x + w/2, Si['ST'], w, label='Total-order $S_T$', color='#534AB7')
    ax.set_xticks(x); ax.set_xticklabels(problem['names'], rotation=20)
    ax.set_ylabel('Sobol index')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.5)

fig.suptitle(f'Sobol global sensitivity (Fc-fusion, {n_runs} runs)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/sobol_indices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/sobol_indices.png and results/tables/sobol_indices.csv")
