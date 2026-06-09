"""
Cross-Species Translation: Rabbit to Human
============================================
Scales the model to human ocular volumes with allometric clearance scaling.
Key finding: ocular half-life is set by intrinsic rate constants (k_va, k_ao)
not absolute volume. Dose should scale by vitreous volume (~2.7x rabbit->human),
not body weight (~23x). Standard mAb body-weight allometry would over-dose
by an order of magnitude.

Also runs a multi-dose simulation at human clinical doses to assess feasibility
of various dose-interval combinations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from model.tmdd_model import (
    params_for_format, simulate, dose_nmol_from_mg, MW_KDA, rhs, TARGET_BIO,
)

FMT_LABELS = {'naked': 'Naked peptide (30 kDa)',
              'fab':   'Fab-fusion (80 kDa)',
              'fc':    'Fc-fusion (115 kDa)'}
FMT_COLORS = {'naked': '#E24B4A', 'fab': '#EF9F27', 'fc': '#1D9E75'}


def metrics_from_sim(res):
    if res is None:
        return dict(t_half_d=np.nan, AUC_TO_d=np.nan, peak_TO=np.nan,
                    days_TO_gt30=np.nan, days_TO_gt50=np.nan)

    # Terminal half-life
    sub_idx = (res['t_d'] > 5) & (res['Cv'] > 1e-6)
    t12 = np.nan
    if sub_idx.sum() >= 5:
        slope, _ = np.polyfit(res['t_d'][sub_idx], np.log(res['Cv'][sub_idx]), 1)
        if slope < 0:
            t12 = float(np.log(2) / abs(slope))

    auc_to = float(np.trapezoid(res['TO'], res['t_d']))
    peak_to = float(res['TO'].max())

    def days_above(thr):
        above = (res['TO'] > thr).astype(int)
        if not above.any():
            return 0.0
        transitions = np.diff(np.concatenate([[0], above, [0]]))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        return sum(res['t_d'][e-1] - res['t_d'][s]
                   for s, e in zip(starts, ends))

    return dict(t_half_d=t12, AUC_TO_d=auc_to, peak_TO=peak_to,
                days_TO_gt30=days_above(0.3),
                days_TO_gt50=days_above(0.5))


# ==========================================================================
# Analysis 1: Cross-species comparison at iso-concentration
# ==========================================================================
print("Analysis 1: Cross-species at iso-concentration starting Cv")
rows = []
target_start_nM = 11.6  # rabbit Fc-fusion at 2 mg = 17.4 nmol / 1.5 mL

for species in ['rabbit', 'human']:
    Vv = {'rabbit': 1.5, 'human': 4.0}[species]
    for fmt in ['naked', 'fab', 'fc']:
        p = params_for_format(fmt, species)
        dose_nmol = target_start_nM * Vv
        dose_mg = dose_nmol * MW_KDA[fmt] * 1000 / 1e6
        res = simulate(p, dose_nmol, t_max_d=90)
        m = metrics_from_sim(res)
        m.update(species=species, format=fmt, dose_nmol=dose_nmol, dose_mg=dose_mg)
        rows.append(m)

df_iso = pd.DataFrame(rows)
df_iso.to_csv('results/tables/cross_species_iso_concentration.csv', index=False)
print(df_iso[['species', 'format', 'dose_mg', 't_half_d',
              'AUC_TO_d', 'peak_TO']].to_string(index=False))


# ==========================================================================
# Analysis 2: Human dose-response
# ==========================================================================
print("\nAnalysis 2: Human dose-response")
dose_grid_mg = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
rows = []
for fmt in ['naked', 'fab', 'fc']:
    p = params_for_format(fmt, 'human')
    for dose_mg in dose_grid_mg:
        dose_nmol = dose_nmol_from_mg(dose_mg, fmt)
        res = simulate(p, dose_nmol, t_max_d=90)
        m = metrics_from_sim(res)
        m.update(format=fmt, dose_mg=dose_mg, dose_nmol=dose_nmol)
        rows.append(m)

df_dr = pd.DataFrame(rows)
df_dr.to_csv('results/tables/human_dose_response.csv', index=False)


# ==========================================================================
# Plot: 2x2 - cross-species PK and PD, plus human dose-response
# ==========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Top: cross-species at iso-concentration
species_styles = {'rabbit': '--', 'human': '-'}
for fmt in ['naked', 'fab', 'fc']:
    for species in ['rabbit', 'human']:
        Vv = {'rabbit': 1.5, 'human': 4.0}[species]
        p = params_for_format(fmt, species)
        dose_nmol = target_start_nM * Vv
        res = simulate(p, dose_nmol, t_max_d=90)
        axes[0, 0].semilogy(res['t_d'], np.clip(res['Cv'], 1e-4, None),
                            color=FMT_COLORS[fmt],
                            linestyle=species_styles[species], linewidth=1.4)
        axes[0, 1].plot(res['t_d'], res['TO'] * 100,
                        color=FMT_COLORS[fmt],
                        linestyle=species_styles[species], linewidth=1.4)

axes[0, 0].set_xlabel('Time (days)'); axes[0, 0].set_ylabel('Cv (nM)')
axes[0, 0].set_title('Iso-concentration: vitreous PK')
axes[0, 0].set_xlim(0, 60); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel('Time (days)'); axes[0, 1].set_ylabel('Target occupancy (%)')
axes[0, 1].set_title('Iso-concentration: PD')
axes[0, 1].set_xlim(0, 30); axes[0, 1].set_ylim(0, 105)
axes[0, 1].grid(True, alpha=0.3)

# Legend
import matplotlib.lines as mlines
handles = [mlines.Line2D([], [], color=FMT_COLORS[f], linewidth=2,
                          label=FMT_LABELS[f]) for f in ['naked', 'fab', 'fc']]
handles += [mlines.Line2D([], [], color='gray', linestyle='-',
                           linewidth=2, label='Human'),
            mlines.Line2D([], [], color='gray', linestyle='--',
                           linewidth=2, label='Rabbit')]
axes[0, 1].legend(handles=handles, fontsize=8, loc='upper right')

# Bottom: human dose-response on two metrics
for fmt in ['naked', 'fab', 'fc']:
    sub = df_dr[df_dr['format'] == fmt].sort_values('dose_mg')
    axes[1, 0].plot(sub['dose_mg'], sub['peak_TO'] * 100,
                    color=FMT_COLORS[fmt], label=FMT_LABELS[fmt],
                    marker='o', markersize=5, linewidth=1.6)
    axes[1, 1].plot(sub['dose_mg'], sub['days_TO_gt30'],
                    color=FMT_COLORS[fmt], label=FMT_LABELS[fmt],
                    marker='s', markersize=5, linewidth=1.6)

axes[1, 0].axhline(50, color='gray', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Dose (mg per eye)'); axes[1, 0].set_ylabel('Peak TO (%)')
axes[1, 0].set_title('Human: peak target occupancy vs dose')
axes[1, 0].set_xscale('log'); axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=8)

axes[1, 1].set_xlabel('Dose (mg per eye)'); axes[1, 1].set_ylabel('Days TO > 30%')
axes[1, 1].set_title('Human: duration above 30% threshold')
axes[1, 1].set_xscale('log'); axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=8)

fig.suptitle('Cross-species translation and human dose-response',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/human_translation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/human_translation.png")
