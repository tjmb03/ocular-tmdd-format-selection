"""
k_off Sweep: Why Slow Off-Rates Don't Rescue Sustained PD
==========================================================
Sweeps k_off across three orders of magnitude (affinity-maturation scenario,
k_on held constant). Headline finding: peak TO plateaus once k_off < k_int,
because complex internalization (not dissociation) is the rate-limiting sink.

Implication: affinity maturation alone cannot deliver durable target engagement.
The mechanism must tolerate transient occupancy or k_int must be engineered down.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from model.tmdd_model import (
    params_for_format, simulate, dose_nmol_from_mg, rhs, TARGET_BIO,
)

FMT_LABELS = {'naked': 'Naked peptide (30 kDa)',
              'fab':   'Fab-fusion (80 kDa)',
              'fc':    'Fc-fusion (115 kDa)'}
FMT_COLORS = {'naked': '#E24B4A', 'fab': '#EF9F27', 'fc': '#1D9E75'}


def simulate_multidose(p, dose_nmol, n_doses=5, tau_d=56, post_d=60):
    """Multi-dose simulation with bolus events at intervals of tau_d days."""
    t_max = n_doses * tau_d + post_d
    t_eval = np.linspace(0, t_max * 24, 3000)
    y = [0, 0, 0, TARGET_BIO['R0_nM'] * p['Vr'], 0, 0]
    t_now = 0
    t_segs, y_segs = [], []
    for d_idx in range(n_doses):
        y[0] += dose_nmol
        t_end = ((d_idx + 1) * tau_d * 24
                 if d_idx < n_doses - 1 else t_max * 24)
        t_seg = t_eval[(t_eval >= t_now) & (t_eval <= t_end)]
        if len(t_seg) < 2:
            continue
        sol = solve_ivp(rhs, (t_now, t_end), y, args=(p,),
                        t_eval=t_seg, method='LSODA', rtol=1e-7, atol=1e-10)
        if not sol.success:
            return None
        t_segs.append(sol.t); y_segs.append(sol.y)
        y = list(sol.y[:, -1]); t_now = t_end
    t_full = np.concatenate(t_segs)
    y_full = np.concatenate(y_segs, axis=1)
    df = pd.DataFrame({
        'time_d': t_full / 24,
        'Cv': y_full[0] / p['Vv'], 'R': y_full[3] / p['Vr'],
        'DR': y_full[4] / p['Vr'],
    })
    df['TO'] = df['DR'] / (df['R'] + df['DR'] + 1e-30)
    return df


# ==========================================================================
# Sweep: k_off varies, k_on held constant (affinity maturation scenario)
# ==========================================================================
k_off_grid = np.logspace(-3, 0, 18)
DOSE_MG_HUMAN = 5.0
DOSE_INTERVAL_D = 56
n_doses = 5

print(f"Sweeping k_off from {k_off_grid[0]:.3f} to {k_off_grid[-1]:.3f} /h")
print(f"Human, {DOSE_MG_HUMAN} mg, q{DOSE_INTERVAL_D}d, {n_doses} doses")

rows = []
trajectories = {}
for fmt in ['naked', 'fab', 'fc']:
    p_def = params_for_format(fmt, 'human')
    k_on_fixed = p_def['k_on']
    dose_nmol = dose_nmol_from_mg(DOSE_MG_HUMAN, fmt)
    trajectories[fmt] = {}

    for k_off in k_off_grid:
        p = params_for_format(fmt, 'human')
        p['k_off'] = k_off

        df_md = simulate_multidose(p, dose_nmol, n_doses=n_doses,
                                    tau_d=DOSE_INTERVAL_D)
        if df_md is None:
            continue

        peak_TO = float(df_md['TO'].max())
        troughs = []
        for d_idx in range(1, n_doses):
            t_just_before = d_idx * DOSE_INTERVAL_D - 0.05
            idx = (df_md['time_d'] - t_just_before).abs().idxmin()
            troughs.append(df_md.loc[idx, 'TO'])

        ss_win = ((df_md['time_d'] >= (n_doses-1)*DOSE_INTERVAL_D) &
                  (df_md['time_d'] <= n_doses*DOSE_INTERVAL_D))
        rows.append(dict(
            format=fmt, k_off_h=k_off, KD_nM=k_off / k_on_fixed,
            peak_TO=peak_TO, trough_TO_mean=float(np.mean(troughs)),
            ss_mean_TO=float(df_md.loc[ss_win, 'TO'].mean()),
        ))
        if np.isclose(k_off, k_off_grid[0]) or np.isclose(k_off, k_off_grid[-1]) \
                or k_off in [k_off_grid[5], k_off_grid[10]]:
            trajectories[fmt][k_off] = df_md

df_sweep = pd.DataFrame(rows)
df_sweep.to_csv('results/tables/koff_sweep.csv', index=False)


# ==========================================================================
# Plot
# ==========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for fmt in ['naked', 'fab', 'fc']:
    sub = df_sweep[df_sweep['format'] == fmt].sort_values('k_off_h')
    axes[0].plot(sub['k_off_h'], sub['peak_TO'] * 100, 'o-',
                 color=FMT_COLORS[fmt], linewidth=1.6, markersize=4,
                 label=FMT_LABELS[fmt])
    axes[1].plot(sub['k_off_h'], sub['trough_TO_mean'] * 100, 's-',
                 color=FMT_COLORS[fmt], linewidth=1.6, markersize=4)
    axes[2].plot(sub['k_off_h'], sub['ss_mean_TO'] * 100, '^-',
                 color=FMT_COLORS[fmt], linewidth=1.6, markersize=4)

ref = {'Engineered slow-off': 0.001, 'Affinity-matured': 0.01,
       'Typical mAb': 0.05, 'Placeholder': 0.25}
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel(r'k$_{off}$ (h$^{-1}$, k$_{on}$ fixed)')
    ax.grid(True, alpha=0.3)
    ax.axhline(30, color='red', linestyle='--', alpha=0.3)
    for lab, k in ref.items():
        ax.axvline(k, color='gray', linestyle=':', alpha=0.4)
axes[0].set_ylabel('Peak TO (%)'); axes[0].set_title('Peak target occupancy')
axes[0].legend(fontsize=8)
axes[1].set_ylabel('Mean trough TO (%)')
axes[1].set_title('Trough TO before each q8w dose')
axes[2].set_ylabel('Steady-state mean TO (%)')
axes[2].set_title('Steady-state TO over q8w window')

y_top = axes[0].get_ylim()[1] * 0.95
for lab, k in ref.items():
    axes[0].text(k, y_top, lab, rotation=90, ha='right', va='top',
                 fontsize=7, color='gray')

fig.suptitle('k_off sweep (k_on fixed, affinity-maturation scenario)\n'
             '5 mg q8w in human - peak TO plateaus once k_off < k_int',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/koff_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/koff_sweep.png")
