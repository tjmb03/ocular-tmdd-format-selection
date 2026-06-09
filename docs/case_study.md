# Case Study Summary

This document distills the key findings, methods, and program implications from the full analysis. For a runnable end-to-end pipeline, see `run_all.py`. For methodology depth, see `docs/methods.md`. For program-level interpretation, see `docs/interpretation.md`.

## One-paragraph summary

A four-compartment intravitreal TMDD model compares three molecular formats (naked peptide ~30 kDa, Fab-fusion ~80 kDa, Fc-fusion ~115 kDa) of a hypothetical microglial target ligand. Calibrated against published rabbit ocular PK (Park 2016) and analyzed via Sobol global sensitivity and a three-phase identifiability analysis (structural via Lie derivatives, profile likelihood, and ABC-SMC Bayesian shrinkage). The model identifies which biology measurements have the highest leverage on first-in-human dose decisions.

## Key findings

**1. Format ranking inverts when controlling for molar dose.** At equal mg, the naked peptide wins on molar count (2 mg of a 30 kDa peptide is 4× more molecules than 2 mg of a 115 kDa Fc-fusion). At equimolar dose, the Fc-fusion delivers ~75% more integrated target engagement, driven by FcRn-mediated retinal residence extension. The program decision therefore depends on whether the binder can be matured to achieve target potency at a lower molar dose.

**2. Target abundance R₀ is the dominant uncertainty.** Sobol total-order sensitivity index ~0.48, the highest of any parameter, with vitreous-to-retina diffusion (k_vr) close behind (~0.43). Measuring R₀ in iPSC-derived microglia is among the highest-leverage experiments.

**3. Internalization, not dissociation, limits sustained PD.** Sweeping k_off across three orders of magnitude shows peak target occupancy plateauing once k_off < k_int. Affinity maturation alone cannot deliver durable target engagement.

**4. Dose should scale by vitreous volume (~2.7×), not body weight (~23×).** Standard mAb body-weight allometry would over-dose by an order of magnitude.

**5. Three identifiability methods give a layered picture.** The structural rank test shows that with vitreous + target-occupancy observations, 11 of 12 parameters are structurally identifiable (only CL_p is not, because plasma is unobserved). Profile likelihood on the four mechanistically key parameters shows k_off has the widest confidence interval of the set (binding off-rate trades off with internalization). ABC-SMC concentrates the posterior on the true values (all 12 posterior means within ~2× of truth); the binding and turnover parameters show the most prior-to-posterior spread, while diffusion parameters (k_va, k_vr) are tightly pinned. At this particle count the credible intervals are approximate — scale up for tighter coverage.

## Method stack

| Layer | Tool | Purpose |
|---|---|---|
| Forward ODE | rxode2 (R) + scipy LSODA (Python) | 6-state TMDD simulation |
| Sensitivity | SALib Sobol indices | Global variance decomposition |
| Structural identifiability | JAX Lie-derivative rank test | Tests parameter recoverability in principle |
| Practical identifiability | Profile likelihood (Raue 2009) | Tests recoverability at realistic data quality |
| Bayesian inference | PyABC ABC-SMC | Posterior shrinkage |
| Production fallback | Stan + cmdstanpy | HMC for production environments |

## Program implications

For a program developing an intravitreal biologic targeting a microglial receptor, this analysis recommends:

- **Measure target abundance R₀ in iPSC-derived microglia first** — highest sensitivity-index contribution
- **Include target occupancy in the preclinical readout** — rescues 8 of 12 parameters from structural non-identifiability
- **Do not invest in precise aqueous outflow characterization** — sensitivity-index ~0
- **Scale FIH dose by vitreous volume**, not body weight
- **Plan for transient receptor engagement** — sustained between-dose target occupancy is unlikely at clinical doses given current target biology assumptions

## Limitations to acknowledge

Five honest model assumptions worth stress-testing:

1. The k_off priors are placeholders (0.25–0.5 /h); real binders may be much slower
2. Single retina compartment doesn't resolve inner-vs-outer retina
3. Cross-species target conservation is assumed (probably wrong at the margin)
4. No target upregulation feedback under chronic dosing
5. The mAb universal-K anchor for CL_p assumes native IgG1 Fc (not engineered variants like YTE/LS)

See `docs/interpretation.md` for the full discussion.
