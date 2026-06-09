"""
Dose-Response Comparison: Equal-mg vs Equimolar
================================================
The headline result of the case study. At equal mg dose the smaller naked
peptide appears best (more molecules per mg). At equimolar dose the Fc-fusion
wins by ~75% on AUC of target occupancy, driven by FcRn-mediated retinal
residence extension.

The implication: the program decision depends on whether the binder can be
matured to achieve target potency at a lower molar dose - not on simple
"which format is better" reasoning.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.tmdd_model import (
    params_for_format, simulate, dose_nmol_from_mg, MW_KDA,
)

FMT_LABELS = {'naked': 'Naked peptide (30 kDa)',
              'fab':   'Fab-fusion (80 kDa)',
              'fc':    'Fc-fusion (115 kDa)'}
FMT_COLORS = {'naked': '#E24B4A', 'fab': '#EF9F27', 'fc': '#1D9E75'}


def metrics_from_sim(res):
    if res is None:
        return dict(peak_TO=np.nan, AUC_TO_d=np.nan)
    return dict(
        peak_TO=float(res['TO'].max()),
        AUC_TO_d=float(np.trapezoid(res['TO'], res['t_d'])),
    )


# ==========================================================================
# Run both regimens for all three formats (rabbit)
# ==========================================================================
equimolar_nmol = dose_nmol_from_mg(2.0, 'fc')   # ~17.4 nmol
print(f"Equimolar reference dose: {equimolar_nmol:.1f} nmol")

rows = []
for fmt in ['naked', 'fab', 'fc']:
    p = params_for_format(fmt, 'rabbit')

    # Equal-mg arm: 2 mg of each
    dose_eqmg = dose_nmol_from_mg(2.0, fmt)
    res_eqmg = simulate(p, dose_eqmg)
    m = metrics_from_sim(res_eqmg)
    m.update(format=fmt, regimen='equal_mg', dose_nmol=dose_eqmg, dose_mg=2.0)
    rows.append(m)

    # Equimolar arm: 17.4 nmol of each
    res_eqmol = simulate(p, equimolar_nmol)
    m = metrics_from_sim(res_eqmol)
    m.update(format=fmt, regimen='equimolar', dose_nmol=equimolar_nmol,
             dose_mg=equimolar_nmol * MW_KDA[fmt] * 1000 / 1e6)
    rows.append(m)

df = pd.DataFrame(rows)
df.to_csv('results/tables/dose_regimen_comparison.csv', index=False)
print("\nResults:")
print(df.to_string(index=False))


# ==========================================================================
# Plot: side-by-side TO curves
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

for ax_idx, regimen in enumerate(['equal_mg', 'equimolar']):
    ax = axes[ax_idx]
    for fmt in ['naked', 'fab', 'fc']:
        p = params_for_format(fmt, 'rabbit')
        dose = dose_nmol_from_mg(2.0, fmt) if regimen == 'equal_mg' \
               else equimolar_nmol
        res = simulate(p, dose)
        ax.plot(res['t_d'], res['TO'] * 100,
                color=FMT_COLORS[fmt], label=FMT_LABELS[fmt], linewidth=1.8)

    ax.axhline(50, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('Time (days)')
    ax.set_xlim(0, 30)
    ax.set_title('Equal mg dose (2 mg each)' if regimen == 'equal_mg'
                 else f'Equimolar dose ({equimolar_nmol:.1f} nmol each)')
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.set_ylabel('Target occupancy (%)')
        ax.legend(fontsize=9, loc='upper right')

fig.suptitle('PD comparison under two dosing regimens',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/equimolar_inversion.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved results/plots/equimolar_inversion.png")
print("(This is the hero image used in the README)")
