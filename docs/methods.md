# Methods

This document describes the model structure, parameter priors, calibration, and analytical methods in enough detail that a reviewer can find any equation or literature anchor without reading the code.

## Model structure

A four-compartment ocular pharmacokinetic / pharmacodynamic model with full Mager-Jusko target-mediated drug disposition (TMDD) localized to the retina compartment.

### Compartments

1. **Vitreous** — dosing site; drug only (no target)
2. **Aqueous humor** — anterior elimination route (~75–85% of total elimination per del Amo Maurice-plot analysis)
3. **Retina / microglia** — free drug, free target, drug-target complex; site of TMDD and FcRn recycling
4. **Systemic plasma** — escape compartment receiving anterior outflow

### State variables (amounts, nmol)

| State | Symbol | Compartment |
|---|---|---|
| A_Cv | vitreous drug | vitreous |
| A_Ca | aqueous drug | aqueous humor |
| A_Cr | retinal free drug | retina |
| A_R | free target | retina |
| A_DR | drug-target complex | retina |
| A_Cp | plasma drug | systemic |

Concentrations are derived as amount divided by compartment volume (e.g., `Cv = A_Cv / Vv` in nM = nmol/mL).

### ODE system

**Vitreous** — free drug, no target:

```
dA_Cv/dt = -k_va · A_Cv + k_av · A_Ca - k_vr · A_Cv + k_rv · A_Cr
```

**Aqueous humor** — receives from vitreous, exits via trabecular outflow:

```
dA_Ca/dt = k_va · A_Cv - k_av · A_Ca - k_ao · A_Ca
```

**Retina free drug** — receives from vitreous, undergoes target binding, FcRn recycling rescues a fraction of internalized complex:

```
dA_Cr/dt = k_vr · A_Cv - k_rv · A_Cr - (k_on · Cr · R - k_off · DR) · Vr
         + phi_FcRn · f_rec · k_int · DR · Vr
```

**Free target** — turnover at constant rate plus binding flux:

```
dA_R/dt = (k_syn - k_deg · R) · Vr - (k_on · Cr · R - k_off · DR) · Vr
```

**Drug-target complex** — formation, dissociation, and internalization:

```
dA_DR/dt = (k_on · Cr · R - k_off · DR) · Vr - k_int · A_DR
```

**Systemic plasma** — anterior spillover, first-order clearance:

```
dA_Cp/dt = k_ao · A_Ca - (CL_p / Vp) · A_Cp
```

The FcRn switch `phi_FcRn ∈ {0, 1}` activates the recycling term only for the Fc-fusion format. `f_rec` is the fraction of internalized complex rescued back to free drug in the retina compartment — a lumped representation of the endosomal salvage pathway, which would otherwise require an explicit endosomal sub-compartment with pH-dependent FcRn binding.

### Why full TMDD, not QSS or QE

The model retains the full Mager-Jusko TMDD structure rather than collapsing to the quasi-steady-state (QSS) or quasi-equilibrium (QE) approximations. The reason is methodological: QSS collapses k_on and k_off into a single composite parameter (K_ss = (k_off + k_int) / k_on), which would prevent mechanistic attribution of any observed format differences. Since the Fc-fusion's hypothesized advantage comes from avidity-enhanced apparent k_off (slower dissociation via re-binding), and the monovalent formats differ only in k_off, the un-collapsed form is required to test the mechanistic hypothesis. The cost is more parameters and harder identifiability — addressed by the identifiability analysis in Phase A–C below.

## Parameter values and priors

All parameters are positive. Lognormal priors are used for rate constants; the beta distribution is used for f_rec because it's bounded in [0, 1].

### Format-specific parameters

| Parameter | Symbol | Naked (~30 kDa) | Fab-fusion (~80 kDa) | Fc-fusion (~115 kDa) | Source |
|---|---|---|---|---|---|
| Vitreous → aqueous | k_va | 0.014 /h | 0.010 /h | 0.009 /h | Stokes-Einstein scaling from Park 2016 rabbit half-lives |
| Aqueous → vitreous backflow | k_av | 0.05 /h | 0.03 /h | 0.02 /h | Symmetric scaling, small backflow |
| Aqueous → outflow | k_ao | 0.40 /h (rabbit), 0.60 /h (human) | same | same | Bulk fluid flow (~2 µL/min ÷ aqueous volume); format-independent |
| Vitreous → retina | k_vr | 0.0025 /h | 0.0015 /h | 0.0013 /h | Posterior diffusion, ~10–20% of k_va |
| Retina → vitreous backflow | k_rv | 0.005 /h | 0.003 /h | 0.0025 /h | Symmetric scaling |
| Target on-rate | k_on | 0.05 /(nM·h) | 0.05 /(nM·h) | 0.10 /(nM·h) | Avidity factor ~2× for bivalent Fc-fusion |
| Target off-rate | k_off | 0.50 /h | 0.50 /h | 0.25 /h | Avidity-slowed apparent k_off for bivalent format |
| Complex internalization | k_int | 0.10 /h | 0.10 /h | 0.10 /h | Receptor-driven, format-independent |
| Target degradation | k_deg | 0.05 /h | 0.05 /h | 0.05 /h | Receptor turnover ~14 h half-life |
| Target synthesis | k_syn | 0.25 nM/h | 0.25 nM/h | 0.25 nM/h | k_syn = k_deg × R₀ (steady state) |
| FcRn recycling fraction | f_rec | 0 | 0 | 0.50 | Active only for Fc-fusion |
| Plasma clearance | CL_p (mL/h/kg) | ~6 | ~1.5 | ~0.19 | Betts 2018 anchor for Fc-fusion |

### Bayesian priors

For Bayesian inference (Phase C), priors are lognormal centered on the format-specific point estimate with moderate sigma (0.5–0.6 in log space, corresponding to ~50–70% CV):

```
log(k_va)  ~ Normal(log(0.009),  0.5)
log(k_av)  ~ Normal(log(0.02),   0.5)
log(k_ao)  ~ Normal(log(0.6),    0.5)
log(k_vr)  ~ Normal(log(0.0013), 0.5)
log(k_rv)  ~ Normal(log(0.0025), 0.5)
log(k_on)  ~ Normal(log(0.10),   0.6)
log(k_off) ~ Normal(log(0.25),   0.6)
log(k_int) ~ Normal(log(0.10),   0.5)
log(k_syn) ~ Normal(log(0.25),   0.5)
log(k_deg) ~ Normal(log(0.05),   0.5)
f_rec      ~ Beta(5, 5)
log(CL_p)  ~ Normal(log(0.57),   0.6)
```

The CL_p prior anchors to Betts et al. 2018 typical mAb clearance (0.15 mL/h/kg, 95% CI 0.14–0.16 mL/h/kg in human), downshifted ~1.5–2× to account for Fc-fusion-specific clearance pathways. For a 3-kg rabbit, this gives CL_p ≈ 0.57 mL/h; for human (70 kg) the allometric scaling factor (70/3)^0.85 ≈ 14.5 gives CL_p ≈ 8.3 mL/h.

## Calibration

The k_va rate constants are anchored to published rabbit vitreous half-lives of three **benchmark molecules** in Park et al. 2016. These are real marketed drugs used as reference points, not the hypothetical model formats:

| Benchmark molecule | Format / size | Reported t½ | Implied k_va (single-compartment) |
|---|---|---|---|
| Ranibizumab | Fab fragment (~48 kDa) | 2.51 d (60 h) | 0.0115 /h |
| Aflibercept | Fc-fusion (~115 kDa) | 3.92 d (94 h) | 0.0074 /h |
| Bevacizumab | Full IgG (~149 kDa) | 6.99 d (168 h) | 0.0041 /h |

The three **model formats** (naked, Fab-fusion, Fc-fusion) are separate hypothetical molecules whose k_va values are set by Stokes-Einstein scaling (k_va ∝ 1/r ∝ 1/MW^(1/3)) to fall within the range bracketed by these benchmarks. They are intentionally not identical to the benchmark drugs.

A note on multi-compartment half-life: the "implied k_va" column above is the single-compartment value that would reproduce each benchmark's terminal half-life if vitreous-to-aqueous diffusion were the only elimination route. In the full four-compartment model, the vitreous is also drained by the posterior route (k_vr) and the drug is then eliminated downstream, so the **model-predicted terminal half-life is shorter than 1/k_va alone would give**. With the model formats' parameters, the predicted rabbit vitreous half-lives are 2.06 d (naked), 2.67 d (Fab-fusion, +6.5% vs ranibizumab's 2.51 d), and 2.95 d (Fc-fusion, −25% vs aflibercept's 3.92 d). The Fc-fusion model format is deliberately faster-clearing than aflibercept because it uses a lower k_va (0.0090 vs aflibercept's effective 0.0074) plus an explicit posterior-route drain — the model format is not meant to be a digital twin of aflibercept. Anterior elimination fraction in the simulated rabbit eye is 78–87% across formats, consistent with the del Amo et al. 2017 Maurice-plot range of 51–85%.

## Analytical methods

### Sobol global sensitivity (Phase 2)

896 model evaluations on six parameters (k_va, k_ao, k_vr, f_rec, k_on, R₀) with log-uniform priors spanning the physiologically plausible range. Output is AUC of target occupancy over 60 days at a 2 mg IVT dose. Both first-order (S₁) and total-order (S_T) indices computed. Implementation: SALib Sobol sampler with `calc_second_order=True`.

### Cross-species translation

Ocular volumes from del Amo & Urtti 2015 / Park 2016: rabbit (Vv = 1.5 mL, Va = 0.30 mL, Vr = 0.25 mL), human (Vv = 4.0 mL, Va = 0.25 mL, Vr = 0.40 mL). Plasma clearance scaled by body weight to the 0.85 power (Betts 2018 allometric exponent for cynomolgus-to-human; comparable for rabbit-to-human within the literature range). The drug-intrinsic diffusion rate constants (k_va, k_vr and their back-flows) are assumed conserved across species, since they reflect molecular size and tissue permeability rather than eye geometry. The aqueous outflow rate k_ao reflects bulk aqueous turnover and is set per species (0.40 /h rabbit, 0.60 /h human); it is not drug-dependent.

### Structural identifiability (Phase A)

Lie-derivative observability rank test, also known as the SIAN methodology (Hong et al. 2019). For a system dx/dt = f(x, p, u) with observations y = h(x, p), the observability matrix is built from Jacobians of the iterated Lie derivatives L_f^k h with respect to parameters:

```
O = [ ∂h/∂p, ∂L_f h/∂p, ∂L_f² h/∂p, ... ]
```

Local structural identifiability requires the rank of O to equal the number of parameters. Implementation uses JAX automatic differentiation (`jacfwd`) to compute Lie derivatives exactly; the rank is computed via SVD with adaptive numerical tolerance. Two observation scenarios are compared: vitreous concentration alone (Scenario A) vs. vitreous + target occupancy (Scenario B).

### Profile likelihood (Phase B)

Raue et al. 2009 framework. Synthetic IVT data are generated at the truth parameter values with 7 timepoints (0.5, 1, 2, 5, 14, 30, 60 days) and 15% multiplicative log-normal noise on paired Cv + TO observations. For each profiled parameter, a grid of values is fixed and all other parameters re-optimized via L-BFGS-B to recover the conditional maximum likelihood. The 95% confidence interval is the range over which the change in negative log-likelihood remains below the chi-squared 1-DOF threshold (Δχ² = 3.84 / 2 = 1.92).

### ABC-SMC Bayesian (Phase C)

Approximate Bayesian Computation with Sequential Monte Carlo via PyABC (Klinger et al. 2018). Distance function is the sum of squared log-residuals on paired Cv + TO observables:

```
d(sim, obs) = Σ_i [ (log Cv_sim_i - log Cv_obs_i)² + (log TO_sim_i - log TO_obs_i)² ]
```

400 particles per generation; up to 9 generations; adaptive epsilon schedule (quantile-based); parallel particle simulation via `MulticoreEvalParallelSampler`. The final particle population approximates the joint posterior. Posterior shrinkage per parameter is computed as `1 - Var(posterior) / Var(prior)`.

## Validation

The Python and R implementations produce identical trajectories to machine precision for the same parameter values (verified by comparing rxode2 output to scipy.integrate.solve_ivp output with LSODA solver, rtol 1e-8, atol 1e-11). Synthetic-data parameter recovery: ABC-SMC concentrates the posterior on the truth, with all 12 posterior means within roughly a factor of two of their true values. (At these particle counts the 95% credible intervals are approximate and a few may not cover the truth on a given run; this is expected for a likelihood-free method. Scale up for tighter coverage.) The robust, reproducible identifiability findings are (i) the structural result that CL_p alone is non-identifiable from vitreous + target-occupancy data, and (ii) the profile-likelihood result that the binding off-rate k_off has the widest confidence interval of the profiled parameters. The fine-grained ABC-SMC shrinkage ranking among well-constrained parameters carries Monte Carlo noise and is not relied upon for specific claims.

## Limitations

Five honest model assumptions that are not directly tested by this analysis:

1. **k_off priors are placeholders** (0.25–0.5 /h). Real binders to specific microglial targets may be much slower; if k_off ≈ 0.005–0.05 /h (typical for affinity-matured binders), the sustained-PD picture changes substantially.
2. **The retina compartment is single-volume**, not split into RPE / photoreceptor / inner-retina sublayers. For posterior segment indications targeting inner-retinal microglia, adding spatial resolution would change diffusion gradients but not the format-comparison conclusions.
3. **Target biology is assumed conserved across species.** Microglial receptor abundance and turnover in human iPSC-derived microglia may differ from rabbit ocular tissue.
4. **No target upregulation feedback.** Disease biology may upregulate or downregulate target with chronic engagement (tachyphylaxis or sensitization).
5. **The mAb universal K anchor for CL_p assumes IgG1 Fc with native FcRn binding.** If the production construct includes Fc engineering (YTE, LS), priors should shift toward longer t½.
