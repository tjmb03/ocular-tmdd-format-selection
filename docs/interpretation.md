# Interpretation: What the Model Says About the Program

This document translates the model outputs into program-relevant guidance. Where `methods.md` answers "how does the model work," this document answers "what should the team do differently because of it."

## The headline result

For an intravitreal biologic targeting a microglial receptor, the molecular-format choice (naked peptide vs. Fab-fusion vs. Fc-fusion) depends critically on how dose is specified — and the answer differs from what intuition suggests.

At **equal mg dose**, smaller formats appear to deliver higher peak target occupancy. This is purely a molar-count artifact: 2 mg of a 30 kDa peptide is 4× more molecules than 2 mg of a 115 kDa Fc-fusion.

At **equimolar dose**, the Fc-fusion delivers ~75% more integrated target engagement than the naked peptide, driven by FcRn-mediated extension of retinal residence time.

The program decision therefore hinges on whether the binder can be matured to achieve target potency at a lower molar dose than the alternative formats — not on simple "which format is better" reasoning.

## What experiments to prioritize

The Sobol sensitivity analysis ranks variance contribution to integrated target occupancy:

| Rank | Parameter | S_T | Recommended experiment |
|---|---|---|---|
| 1 | R₀ (target abundance) | 0.48 | iPSC-derived microglia surface staining, quantitative flow cytometry |
| 2 | k_vr (vitreous → retina) | 0.43 | Ex vivo retinal tissue exposure measurement |
| 3 | f_rec (FcRn recycling) | 0.19 | Tg32 mouse PK + paired target occupancy assay |
| 4 | k_on (binding rate) | 0.16 | BLI / SPR on recombinant target |
| 5 | k_va (vitreous → aqueous) | 0.14 | Already constrained by Park 2016-class published data |
| 6 | k_ao (aqueous outflow) | ~0 | **Don't measure precisely.** Bulk flow is format-independent. |

The k_ao result is the negative finding worth highlighting: aqueous outflow rate has essentially zero impact on model output once it exceeds the rate of vitreous-to-aqueous diffusion (always true physiologically). Programs that invest in detailed aqueous flow characterization for this kind of biologic are wasting resources.

## What the identifiability analysis adds

The three-phase identifiability analysis (structural → profile likelihood → ABC-SMC) tells us *which parameters can actually be recovered* from realistic preclinical data, given a paired Cv + TO measurement design.

**Structural (Phase A), deterministic:** With vitreous + target-occupancy observations, 11 of 12 parameters are structurally identifiable. Only **CL_p** is structurally non-identifiable — and for a clear reason: plasma is not observed in either scenario, so systemic clearance cannot be pinned down. This is a model-correctness check, not a surprise, and it directly implies that any program wanting systemic-exposure estimates needs explicit plasma sampling (standard for IVT safety anyway).

**Profile likelihood (Phase B):** Of the four mechanistically key parameters profiled (k_va, k_off, f_rec, k_int), **k_off has the widest 95% confidence interval** — roughly a 2–3× range depending on data realization — because the binding off-rate trades off against internalization (k_int) in setting complex residence time. k_va is essentially pinpointed. Note that profile likelihood (which fixes one parameter and re-optimizes the rest) and ABC marginal shrinkage can disagree for a parameter like f_rec: its profile is narrow near the optimum, but its ABC marginal posterior stays close to the prior because f_rec is only weakly constrained once the other parameters are free to compensate. The two views are complementary, not contradictory — profile likelihood asks "how much does the fit degrade if this parameter moves," while marginal shrinkage asks "how much did the data narrow this parameter's standalone distribution."

**ABC-SMC (Phase C):** The posterior concentrates on the true values — all 12 posterior means fall within roughly a factor of two of the truth. Posterior shrinkage (1 − Var_post/Var_prior) is highest for the diffusion parameters (k_va, k_vr) and lower for binding/turnover parameters. Two honest caveats at modest particle counts: first, the 95% credible intervals are approximate, and a few parameters may not cover the truth on a given run — this is expected for a likelihood-free method at modest particle counts (and even exact 95% intervals would be expected to miss roughly one parameter in twenty by construction). Second, the fine-grained shrinkage *ranking* among the well-constrained parameters carries Monte Carlo noise and shifts run-to-run. The robust, reproducible findings are the structural result (CL_p non-identifiable) and the profile-likelihood result (k_off widest CI). Scale to 1000+ particles for stable coverage and ranking.

**A note on CL_p shrinkage:** ABC may report high apparent shrinkage for CL_p on a given run, but this is an artifact, not data-driven inference. CL_p governs only the unobserved plasma compartment, so the observables (Cv, TO) carry essentially no information about it — the structural analysis (Phase A) confirms CL_p is the one parameter that remains non-identifiable even with target-occupancy data. Any apparent narrowing of the CL_p posterior comes from the SMC perturbation kernel contracting over generations on a parameter the data cannot constrain, not from genuine learning. This is exactly the distinction the shrinkage metric can obscure: shrinkage measures prior-to-posterior variance reduction, which is not the same as data informativeness. Phase A, not the CL_p shrinkage value, is the honest signal here.

## What this means for first-in-human dose

Three implications for the FIH dose-finding plan:

**Dose should scale by vitreous volume, not body weight.** Cross-species translation shows ocular half-life is set by intrinsic rate constants (k_va, k_ao) that are drug properties, not species properties. The human eye is 2.7× larger than the rabbit eye, but plasma volume is 15× larger. Standard mAb body-weight allometry would over-dose by roughly an order of magnitude. The right scaling factor is the vitreous volume ratio (~2.7×) modulated by target abundance × volume.

**Plan plasma sampling alongside IVT PK.** The structural identifiability analysis shows CL_p is the only parameter that cannot be identified from vitreous + target occupancy data. This is a model-correctness check, not a finding — but it does say that any program planning to characterize systemic exposure (e.g., for safety) needs explicit plasma sampling. The good news: that's already standard practice for IVT studies for safety reasons.

**Sustained between-dose target occupancy is unlikely at clinical doses.** Multi-dose human simulation at 5 mg q8w shows trough target occupancy near zero for all formats. The k_off sweep showed this isn't fixable by affinity maturation alone: once k_off drops below k_int (~0.10 /h), complex internalization becomes the rate-limiting sink. The mechanism must therefore tolerate transient receptor engagement — either through downstream signaling persistence or through an alternate dose regimen. This is a program-level question that should be addressed before committing to a Phase 2 dosing scheme.

## A note on the Fc-fusion advantage

The Fc-fusion has a real mechanistic advantage in this model, but the magnitude is **bounded by ocular biology**:

- Anterior elimination via aqueous humor accounts for 78–87% of total drug clearance — and this route is format-independent (bulk fluid flow, not FcRn-mediated)
- Only the 13–22% of drug taking the posterior route into retina sees FcRn recycling
- Even on that route, recycling is fractional (f_rec = 0.50 in the central estimate)

So the Fc-fusion's residence-time advantage is real but modest compared to the systemic Fc benefit familiar from full IgG mAbs. The argument for Fc-fusion development should rest on:

1. **Tighter apparent KD via avidity** (relevant if target is densely arrayed on microglia — depends on R₀)
2. **Predictable systemic PK once drug escapes the eye** (relevant for safety profile)
3. **Manufacturing maturity** (an off-model argument, but real)

If any of these are weak — for example, if the binder is intrinsically monovalent or the target is sparse — the Fc-fusion advantage shrinks substantially and a simpler format may be preferable.

## Where the model would be wrong

Three assumptions I'd want stress-tested before using this for a regulatory submission:

1. **The k_off values used here are placeholders** (0.25–0.5 /h). Real binders are often slower (0.005–0.05 /h after affinity maturation). The sustained-PD picture depends strongly on this — re-run with measured BLI/SPR data before drawing dose-interval conclusions.
2. **The retina is treated as a single well-mixed compartment.** For inner-retinal targets (e.g., retinal ganglion cells), drug distribution may be inhomogeneous. Adding an inner-retina sub-compartment would change quantitative predictions but probably not the format ranking.
3. **Target conservation across species.** If human microglial receptor density differs substantially from rodent, R₀ for the FIH model should be measured in human iPSC-derived microglia rather than inferred allometrically.

These limitations are not unique to this model — they're shared by essentially all early-stage QSP work — but they should be acknowledged in any communication of results to non-modelers.
