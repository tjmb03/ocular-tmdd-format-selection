// Ocular TMDD model - Bayesian inference in Stan
// =========================================================
// Step 4 Phase 3 (production-ready version)
//
// Fits the 4-compartment + TMDD + FcRn recycling model to vitreous + TO data.
// Used to compute posterior shrinkage as the third leg of identifiability
// analysis (after structural / Phase 1 and profile likelihood / Phase 2).
//
// Priors:
//   - All rate constants: lognormal, mu = log(literature anchor), sigma per-param
//   - f_rec: beta(5, 5) centered on 0.5
//   - CL_p: lognormal anchored to Betts 2018 typical mAb K, downshifted for Fc-fusion
//
// To run with cmdstanpy:
//   import cmdstanpy
//   model = cmdstanpy.CmdStanModel(stan_file='tmdd_model.stan')
//   fit = model.sample(data=data_dict, chains=4, iter_warmup=1000,
//                       iter_sampling=2000, parallel_chains=4,
//                       adapt_delta=0.95, max_treedepth=12)
//
// Compute budget: ~30-90 min on 4-core workstation (real Stan, real ODE solver).

functions {
  // ODE RHS: 6 states (vitreous, aqueous, retina free, target, complex, plasma)
  // Returns dy/dt as a vector. Stan's ODE interface uses real[] not vectors,
  // so we unpack carefully.
  vector tmdd_rhs(real t, vector y, vector p, data vector phys) {
    real A_Cv = y[1];
    real A_Ca = y[2];
    real A_Cr = y[3];
    real A_R  = y[4];
    real A_DR = y[5];
    real A_Cp = y[6];

    real k_va  = p[1];
    real k_av  = p[2];
    real k_ao  = p[3];
    real k_vr  = p[4];
    real k_rv  = p[5];
    real k_on  = p[6];
    real k_off = p[7];
    real k_int = p[8];
    real k_syn = p[9];
    real k_deg = p[10];
    real f_rec = p[11];
    real CL_p  = p[12];

    real Vv = phys[1];
    real Va = phys[2];
    real Vr = phys[3];
    real Vp = phys[4];
    real phi_FcRn = phys[5];

    real Cr = A_Cr / Vr;
    real R  = A_R  / Vr;
    real DR = A_DR / Vr;

    real bind   = k_on  * Cr * R;
    real unbind = k_off * DR;
    real recycle = phi_FcRn * f_rec * k_int * DR;

    vector[6] dy;
    dy[1] = -k_va*A_Cv + k_av*A_Ca - k_vr*A_Cv + k_rv*A_Cr;
    dy[2] =  k_va*A_Cv - k_av*A_Ca - k_ao*A_Ca;
    dy[3] =  k_vr*A_Cv - k_rv*A_Cr - (bind - unbind)*Vr + recycle*Vr;
    dy[4] = (k_syn - k_deg*R)*Vr - (bind - unbind)*Vr;
    dy[5] = (bind - unbind)*Vr - k_int*A_DR;
    dy[6] =  k_ao*A_Ca - (CL_p/Vp)*A_Cp;
    return dy;
  }
}

data {
  int<lower=1> N;                    // number of observation timepoints
  array[N] real<lower=0> t_obs;      // observation times (hours)
  vector<lower=0>[N] Cv_obs;         // vitreous concentration observations (nM)
  vector<lower=0,upper=1>[N] TO_obs; // target occupancy observations (fraction)
  real<lower=0> dose_nmol;           // IVT bolus dose (nmol)
  real<lower=0> noise_sigma_Cv;      // log-normal noise sigma for Cv
  real<lower=0> noise_sigma_TO;      // log-normal noise sigma for TO

  // Physiological constants (known, not estimated)
  real<lower=0> Vv;
  real<lower=0> Va;
  real<lower=0> Vr;
  real<lower=0> Vp;
  real<lower=0,upper=1> phi_FcRn;    // 1 for Fc-fusion, 0 for others

  // Prior hyperparameters (centers on log scale)
  real prior_log_k_va_mu;   real prior_log_k_va_sd;
  real prior_log_k_av_mu;   real prior_log_k_av_sd;
  real prior_log_k_ao_mu;   real prior_log_k_ao_sd;
  real prior_log_k_vr_mu;   real prior_log_k_vr_sd;
  real prior_log_k_rv_mu;   real prior_log_k_rv_sd;
  real prior_log_k_on_mu;   real prior_log_k_on_sd;
  real prior_log_k_off_mu;  real prior_log_k_off_sd;
  real prior_log_k_int_mu;  real prior_log_k_int_sd;
  real prior_log_k_syn_mu;  real prior_log_k_syn_sd;
  real prior_log_k_deg_mu;  real prior_log_k_deg_sd;
  real prior_f_rec_a;       real prior_f_rec_b;       // beta hyperparameters
  real prior_log_CL_p_mu;   real prior_log_CL_p_sd;
}

transformed data {
  vector[5] phys;
  phys[1] = Vv; phys[2] = Va; phys[3] = Vr; phys[4] = Vp; phys[5] = phi_FcRn;
}

parameters {
  // Sample on log scale for positivity and better NUTS geometry
  real log_k_va;
  real log_k_av;
  real log_k_ao;
  real log_k_vr;
  real log_k_rv;
  real log_k_on;
  real log_k_off;
  real log_k_int;
  real log_k_syn;
  real log_k_deg;
  real<lower=0,upper=1> f_rec;
  real log_CL_p;
}

transformed parameters {
  // Convert back to natural scale
  real<lower=0> k_va  = exp(log_k_va);
  real<lower=0> k_av  = exp(log_k_av);
  real<lower=0> k_ao  = exp(log_k_ao);
  real<lower=0> k_vr  = exp(log_k_vr);
  real<lower=0> k_rv  = exp(log_k_rv);
  real<lower=0> k_on  = exp(log_k_on);
  real<lower=0> k_off = exp(log_k_off);
  real<lower=0> k_int = exp(log_k_int);
  real<lower=0> k_syn = exp(log_k_syn);
  real<lower=0> k_deg = exp(log_k_deg);
  real<lower=0> CL_p  = exp(log_CL_p);

  vector[12] p;
  p[1]=k_va; p[2]=k_av; p[3]=k_ao; p[4]=k_vr; p[5]=k_rv;
  p[6]=k_on; p[7]=k_off; p[8]=k_int; p[9]=k_syn; p[10]=k_deg;
  p[11]=f_rec; p[12]=CL_p;

  // Initial conditions: bolus into vitreous, target at steady state
  real R0 = k_syn / k_deg;
  vector[6] y0;
  y0[1] = dose_nmol;
  y0[2] = 0;
  y0[3] = 0;
  y0[4] = R0 * Vr;
  y0[5] = 0;
  y0[6] = 0;

  // Solve ODE
  // ode_bdf is the recommended stiff solver in Stan
  array[N] vector[6] y_hat
    = ode_bdf_tol(tmdd_rhs, y0, 0.0, t_obs, 1e-6, 1e-9, 10000, p, phys);

  // Predicted observables
  vector[N] Cv_pred;
  vector[N] TO_pred;
  for (i in 1:N) {
    Cv_pred[i] = y_hat[i, 1] / Vv;
    real R_i  = y_hat[i, 4] / Vr;
    real DR_i = y_hat[i, 5] / Vr;
    TO_pred[i] = DR_i / (R_i + DR_i + 1e-12);
    // Numerical safety
    if (Cv_pred[i] < 1e-12) Cv_pred[i] = 1e-12;
    if (TO_pred[i] < 1e-12) TO_pred[i] = 1e-12;
    if (TO_pred[i] > 1 - 1e-12) TO_pred[i] = 1 - 1e-12;
  }
}

model {
  // Priors on log scale
  log_k_va  ~ normal(prior_log_k_va_mu,  prior_log_k_va_sd);
  log_k_av  ~ normal(prior_log_k_av_mu,  prior_log_k_av_sd);
  log_k_ao  ~ normal(prior_log_k_ao_mu,  prior_log_k_ao_sd);
  log_k_vr  ~ normal(prior_log_k_vr_mu,  prior_log_k_vr_sd);
  log_k_rv  ~ normal(prior_log_k_rv_mu,  prior_log_k_rv_sd);
  log_k_on  ~ normal(prior_log_k_on_mu,  prior_log_k_on_sd);
  log_k_off ~ normal(prior_log_k_off_mu, prior_log_k_off_sd);
  log_k_int ~ normal(prior_log_k_int_mu, prior_log_k_int_sd);
  log_k_syn ~ normal(prior_log_k_syn_mu, prior_log_k_syn_sd);
  log_k_deg ~ normal(prior_log_k_deg_mu, prior_log_k_deg_sd);
  f_rec     ~ beta(prior_f_rec_a, prior_f_rec_b);
  log_CL_p  ~ normal(prior_log_CL_p_mu,  prior_log_CL_p_sd);

  // Likelihood: log-normal on both observables
  // log(y_obs) ~ Normal(log(y_pred), sigma)
  for (i in 1:N) {
    target += lognormal_lpdf(Cv_obs[i] | log(Cv_pred[i]), noise_sigma_Cv);
    target += lognormal_lpdf(TO_obs[i] | log(TO_pred[i]), noise_sigma_TO);
  }
}

generated quantities {
  // Posterior predictive samples and log-likelihood for LOO/WAIC if needed
  vector[N] Cv_rep;
  vector[N] TO_rep;
  vector[N] log_lik;
  for (i in 1:N) {
    Cv_rep[i] = lognormal_rng(log(Cv_pred[i]), noise_sigma_Cv);
    TO_rep[i] = lognormal_rng(log(TO_pred[i]), noise_sigma_TO);
    log_lik[i] =
      lognormal_lpdf(Cv_obs[i] | log(Cv_pred[i]), noise_sigma_Cv)
    + lognormal_lpdf(TO_obs[i] | log(TO_pred[i]), noise_sigma_TO);
  }
}
