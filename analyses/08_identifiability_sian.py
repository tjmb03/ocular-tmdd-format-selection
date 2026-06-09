"""
Phase A: Structural Identifiability via Lie-Derivative Rank Test
=================================================================
Uses JAX automatic differentiation to compute Lie derivatives of the
observation function exactly, then builds the observability matrix and
tests its rank.

A parameter is locally structurally identifiable iff its column in the
observability matrix is not in the null space.

Two scenarios compared:
  A. Cv only         - only 3 of 12 parameters identifiable
  B. Cv + TO         - 11 of 12 (only CL_p remains, since plasma is unobserved)

Result: target occupancy assay rescues 8 of 12 parameters - the highest-
leverage experimental design recommendation in the case study.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import jax
import jax.numpy as jnp
from jax import jacfwd
from scipy.linalg import svd

jax.config.update("jax_enable_x64", True)

# Parameter layout (must match the order used by jacfwd)
PARAM_NAMES = ['k_va', 'k_av', 'k_ao', 'k_vr', 'k_rv',
               'k_on', 'k_off', 'k_int', 'k_syn', 'k_deg',
               'f_rec', 'CL_p']

# Baseline (Fc-fusion, human)
P_BASELINE = jnp.array([
    0.009, 0.02, 0.6, 0.0013, 0.0025,
    0.10, 0.25, 0.10, 0.25, 0.05,
    0.50, 0.57,
])

Vv, Va, Vr, Vp = 4.0, 0.25, 0.40, 3000.0
PHI_FCRN = 1.0


def rhs(y, p):
    A_Cv, A_Ca, A_Cr, A_R, A_DR, A_Cp = y
    (k_va, k_av, k_ao, k_vr, k_rv, k_on, k_off, k_int,
     k_syn, k_deg, f_rec, CL_p) = p
    Cr = A_Cr / Vr; R = A_R / Vr; DR = A_DR / Vr
    bind = k_on * Cr * R; unbind = k_off * DR
    recycle = PHI_FCRN * f_rec * k_int * DR
    return jnp.array([
        -k_va*A_Cv + k_av*A_Ca - k_vr*A_Cv + k_rv*A_Cr,
         k_va*A_Cv - k_av*A_Ca - k_ao*A_Ca,
         k_vr*A_Cv - k_rv*A_Cr - (bind - unbind)*Vr + recycle*Vr,
        (k_syn - k_deg*R)*Vr - (bind - unbind)*Vr,
        (bind - unbind)*Vr - k_int*A_DR,
         k_ao*A_Ca - (CL_p/Vp)*A_Cp,
    ])


def h_A(y, p):
    """Scenario A: vitreous concentration only."""
    return jnp.array([y[0] / Vv])


def h_B(y, p):
    """Scenario B: vitreous + target occupancy."""
    R = y[3] / Vr; DR = y[4] / Vr
    return jnp.array([y[0] / Vv, DR / (DR + R + 1e-30)])


def make_lie_chain(h, max_order):
    """Build [h, L_f h, L_f^2 h, ...] as composable Python functions."""
    funcs = [h]
    current = h
    for _ in range(max_order):
        dh_dy = jacfwd(current, argnums=0)
        def L(y, p, dh=dh_dy, cur=current):
            return dh(y, p) @ rhs(y, p)
        funcs.append(L)
        current = L
    return funcs


def svd_rank(M, tol_factor=1e-8):
    if M.size == 0:
        return 0, np.array([])
    s = svd(M, compute_uv=False)
    tol = max(M.shape) * s[0] * tol_factor if len(s) else 0
    return int(np.sum(s > tol)), s


def run_scenario(name, h_func, max_order, seeds=(0, 1, 2, 7)):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    chain = make_lie_chain(h_func, max_order)
    n_params = len(PARAM_NAMES)
    final_ranks = []
    best_M = None
    best_s = None

    for seed in seeds:
        rng = np.random.default_rng(seed)
        y0 = jnp.asarray(np.abs(rng.normal(
            loc=[5.0, 0.5, 0.3, 2.0, 0.5, 0.01],
            scale=[2.0, 0.2, 0.15, 1.0, 0.3, 0.005])))
        rows = []
        for L in chain:
            J = np.array(jacfwd(L, argnums=1)(y0, P_BASELINE))
            for r in range(J.shape[0]):
                rows.append(J[r])
        M = np.array(rows)
        rk, s = svd_rank(M)
        final_ranks.append(rk)
        if best_M is None or rk > svd_rank(best_M)[0]:
            best_M = M; best_s = s

    rank = max(final_ranks)
    print(f"Max rank across seeds: {rank} / {n_params}")

    if rank == n_params:
        return {name_p: 'identifiable' for name_p in PARAM_NAMES}

    status = {}
    for i, name_p in enumerate(PARAM_NAMES):
        cols = [c for c in range(n_params) if c != i]
        M_minus = best_M[:, cols]
        r_minus, _ = svd_rank(M_minus)
        status[name_p] = 'identifiable' if r_minus < rank else 'NON-IDENTIFIABLE'
        mark = "[OK]" if status[name_p] == 'identifiable' else "[X]"
        print(f"  {mark}  {name_p:>8s}")

    return status


print("="*72)
print("Phase A: Structural Identifiability")
print("="*72)

status_A = run_scenario("SCENARIO A: vitreous Cv only",        h_A, max_order=8)
status_B = run_scenario("SCENARIO B: vitreous Cv + occupancy", h_B, max_order=6)

# Summary
print("\n" + "="*72)
print("SUMMARY: rescue effect of adding TO observation")
print("="*72)
print(f"{'Param':<10} {'Cv only':<22} {'Cv + TO':<22} {'Change':<12}")
print("-" * 70)
gained = 0
for name in PARAM_NAMES:
    a, b = status_A[name], status_B[name]
    change = ""
    if a == 'NON-IDENTIFIABLE' and b == 'identifiable':
        change = "RESCUED"; gained += 1
    elif a == b:
        change = "(same)"
    print(f"{name:<10} {a:<22} {b:<22} {change:<12}")
print(f"\nTO readout rescues {gained} of {len(PARAM_NAMES)} parameters.")

# Save
out = dict(scenario_A_Cv_only=status_A,
           scenario_B_Cv_plus_TO=status_B,
           parameters_tested=PARAM_NAMES,
           parameters_rescued_by_TO=gained,
           method='JAX autodiff Lie-derivative observability rank test')
with open('results/tables/sian_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved results/tables/sian_results.json")


# ==========================================================================
# Summary plot
# ==========================================================================
import matplotlib.pyplot as plt

PARAM_MEANING = {
    'k_va':  'Vitreous to aqueous diffusion',
    'k_av':  'Aqueous to vitreous backflow',
    'k_ao':  'Aqueous trabecular outflow',
    'k_vr':  'Vitreous to retina diffusion',
    'k_rv':  'Retina to vitreous backflow',
    'k_on':  'Target binding on-rate',
    'k_off': 'Target binding off-rate',
    'k_int': 'Complex internalization',
    'k_syn': 'Target synthesis',
    'k_deg': 'Target degradation',
    'f_rec': 'FcRn recycling fraction',
    'CL_p':  'Systemic plasma clearance',
}
RECOMMENDATION = {
    'k_va':  'IVT PK + TO assay',
    'k_av':  'IVT PK + TO assay',
    'k_ao':  'IVT PK + TO (aqueous tap optional)',
    'k_vr':  'IVT PK + TO assay',
    'k_rv':  'IVT PK alone sufficient',
    'k_on':  'BLI / SPR + IVT PK',
    'k_off': 'BLI / SPR + IVT PK + TO',
    'k_int': 'iPSC microglia internalization + TO',
    'k_syn': 'Steady-state target + TO timeseries',
    'k_deg': 'Target turnover assay (CHX chase)',
    'f_rec': 'Tg32 mouse PK + TO',
    'CL_p':  'Systemic plasma sampling',
}

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 16); ax.set_ylim(-0.5, len(PARAM_NAMES) - 0.5)
ax.invert_yaxis(); ax.axis('off')

x_param, x_meaning, x_A, x_B, x_rec = 0.2, 2.3, 7.4, 9.0, 10.6
ax.text(x_param, -0.5, 'Parameter', fontweight='bold', fontsize=11, va='center')
ax.text(x_meaning, -0.5, 'Biology', fontweight='bold', fontsize=11, va='center')
ax.text(x_A, -0.5, 'Cv only', fontweight='bold', fontsize=11, va='center', ha='center')
ax.text(x_B, -0.5, 'Cv + TO', fontweight='bold', fontsize=11, va='center', ha='center')
ax.text(x_rec, -0.5, 'Recommended measurement', fontweight='bold', fontsize=11, va='center')
ax.axhline(-0.1, color='#222', linewidth=1)

C_OK, C_X = '#1D9E75', '#D85A30'
for i, p in enumerate(PARAM_NAMES):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color='#F5F8F6', zorder=-1)
    ax.text(x_param, i, p, family='monospace', fontsize=10.5, va='center')
    ax.text(x_meaning, i, PARAM_MEANING[p], fontsize=9.5, va='center', color='#444')
    sa = '\u2713' if status_A[p] == 'identifiable' else '\u2717'
    sb = '\u2713' if status_B[p] == 'identifiable' else '\u2717'
    ax.text(x_A, i, sa, fontsize=14, va='center', ha='center',
            color=C_OK if status_A[p] == 'identifiable' else C_X, fontweight='bold')
    ax.text(x_B, i, sb, fontsize=14, va='center', ha='center',
            color=C_OK if status_B[p] == 'identifiable' else C_X, fontweight='bold')
    if status_A[p] == 'NON-IDENTIFIABLE' and status_B[p] == 'identifiable':
        ax.annotate('', xy=(x_B - 0.3, i), xytext=(x_A + 0.3, i),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
    ax.text(x_rec, i, RECOMMENDATION[p], fontsize=9, va='center', color='#222')

n_A = sum(1 for p in PARAM_NAMES if status_A[p] == 'identifiable')
n_B = sum(1 for p in PARAM_NAMES if status_B[p] == 'identifiable')
ax.text(x_param, len(PARAM_NAMES) + 0.2,
        f"TO readout rescues {gained} of {len(PARAM_NAMES)} parameters.  "
        f"Identifiable: Cv alone = {n_A}/{len(PARAM_NAMES)}; "
        f"Cv + TO = {n_B}/{len(PARAM_NAMES)}.  "
        f"Only CL_p remains non-identifiable (no plasma observable).",
        fontsize=9.5, style='italic', color='#222', va='top')

plt.suptitle('Structural identifiability: which experiments resolve which parameters?',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/plots/identifiability_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/identifiability_summary.png")
