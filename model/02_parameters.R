# ============================================================================
# Format-Specific Parameter Sets for the Ocular TMDD Model
# ----------------------------------------------------------------------------
# Three parameter sets corresponding to the three molecular formats.
# Each parameter has both a point estimate (for deterministic simulation)
# and a prior specification (for Bayesian fitting).
#
# Priors are documented inline with their literature source. See docs/methods.md.
#
# Units:
#   - Concentrations: nM (= nmol/mL)
#   - Volumes:        mL
#   - Rates:          h^-1
#   - Clearances:     mL/h (whole-body, not per-kg, for ocular model)
# ============================================================================

# ----------------------------------------------------------------------------
# Physiological volumes (rabbit reference; see human_translation.py for human)
# ----------------------------------------------------------------------------
phys_volumes_rabbit <- list(
  Vv = 1.5,    # mL, vitreous volume
  Va = 0.30,   # mL, aqueous humor volume
  Vr = 0.25,   # mL, retina + RPE + choroid effective volume
  Vp = 200     # mL, plasma volume (rabbit ~3 kg)
)

phys_volumes_human <- list(
  Vv = 4.0,    # mL, human vitreous
  Va = 0.25,   # mL
  Vr = 0.40,   # mL
  Vp = 3000    # mL, human plasma (~70 kg)
)

# ----------------------------------------------------------------------------
# Target biology (microglial receptor - placeholder values; refine with
# iPSC-microglia screening data when available)
# ----------------------------------------------------------------------------
target_biology <- list(
  R0_nM     = 5,        # baseline target concentration in retina (nM)
  k_deg_h   = 0.05,     # target turnover (h^-1), receptor t1/2 ~14 h
  k_int_h   = 0.10      # complex internalization (h^-1)
)
target_biology$k_syn_nM_h <- target_biology$k_deg_h * target_biology$R0_nM

# ----------------------------------------------------------------------------
# Aqueous outflow rate (bulk fluid flow; format-independent)
# k_ao = aqueous turnover rate / Va
# Rabbit: ~3 uL/min / 0.3 mL    = 0.60 /h (use 0.40 for rabbit per Park 2016)
# Human:  ~2.5 uL/min / 0.25 mL = 0.60 /h
# ----------------------------------------------------------------------------
k_ao_rabbit <- 0.40
k_ao_human  <- 0.60

# ----------------------------------------------------------------------------
# Format 1: Naked peptide (~30 kDa, monovalent, no FcRn)
# ----------------------------------------------------------------------------
params_naked_rabbit <- list(
  Vv = phys_volumes_rabbit$Vv, Va = phys_volumes_rabbit$Va,
  Vr = phys_volumes_rabbit$Vr, Vp = phys_volumes_rabbit$Vp,
  k_va = 0.0140, k_av = 0.05, k_ao = k_ao_rabbit,
  k_vr = 0.0025, k_rv = 0.005,
  k_on = 0.05, k_off = 0.50,
  k_int = target_biology$k_int_h,
  k_syn = target_biology$k_syn_nM_h, k_deg = target_biology$k_deg_h,
  phi_FcRn = 0, f_rec = 0,
  CL_p = 18.0   # mL/h for 3 kg rabbit (~6 mL/h/kg, GFR-limited peptide)
)

# ----------------------------------------------------------------------------
# Format 2: Fab-fusion (~80 kDa, monovalent, no FcRn)
# ----------------------------------------------------------------------------
params_fab_rabbit <- list(
  Vv = phys_volumes_rabbit$Vv, Va = phys_volumes_rabbit$Va,
  Vr = phys_volumes_rabbit$Vr, Vp = phys_volumes_rabbit$Vp,
  k_va = 0.0102, k_av = 0.03, k_ao = k_ao_rabbit,
  k_vr = 0.0015, k_rv = 0.003,
  k_on = 0.05, k_off = 0.50,
  k_int = target_biology$k_int_h,
  k_syn = target_biology$k_syn_nM_h, k_deg = target_biology$k_deg_h,
  phi_FcRn = 0, f_rec = 0,
  CL_p = 4.5    # mL/h for 3 kg rabbit (ranibizumab-like ~1.5 mL/h/kg)
)

# ----------------------------------------------------------------------------
# Format 3: Fc-fusion (~115 kDa, bivalent, FcRn-engaged)
# ----------------------------------------------------------------------------
params_fc_rabbit <- list(
  Vv = phys_volumes_rabbit$Vv, Va = phys_volumes_rabbit$Va,
  Vr = phys_volumes_rabbit$Vr, Vp = phys_volumes_rabbit$Vp,
  k_va = 0.0090, k_av = 0.02, k_ao = k_ao_rabbit,
  k_vr = 0.0013, k_rv = 0.0025,
  k_on = 0.10, k_off = 0.25,
  k_int = target_biology$k_int_h,
  k_syn = target_biology$k_syn_nM_h, k_deg = target_biology$k_deg_h,
  phi_FcRn = 1, f_rec = 0.50,
  CL_p = 0.57   # mL/h for 3 kg rabbit (Betts 2018 mAb K downshifted 1.5x)
)

# ----------------------------------------------------------------------------
# Bayesian prior specifications (for ABC-SMC / Stan / NumPyro)
# ----------------------------------------------------------------------------
priors <- list(
  k_va  = list(dist = "lognormal", meanlog_fn = function(x) log(x), sdlog = 0.5,
               source = "Stokes-Einstein scaling from Park 2016 rabbit data"),
  k_av  = list(dist = "lognormal", sdlog = 0.5),
  k_ao  = list(dist = "lognormal", sdlog = 0.5,
               source = "Bulk fluid outflow; format-independent"),
  k_vr  = list(dist = "lognormal", sdlog = 0.5),
  k_rv  = list(dist = "lognormal", sdlog = 0.5),
  k_on  = list(dist = "lognormal", sdlog = 0.6),
  k_off = list(dist = "lognormal", sdlog = 0.6),
  k_int = list(dist = "lognormal", sdlog = 0.5),
  k_syn = list(dist = "lognormal", sdlog = 0.5),
  k_deg = list(dist = "lognormal", sdlog = 0.5),
  f_rec = list(dist = "beta", a = 5, b = 5,
               source = "Weakly informative, centered on 0.5"),
  CL_p  = list(dist = "lognormal", sdlog = 0.6,
               source = "Betts 2018 typical mAb K = 0.15 mL/h/kg, downshifted 1.5-2x for Fc-fusion")
)

# ----------------------------------------------------------------------------
# Helper: assemble parameter vector for rxode2 simulation
# ----------------------------------------------------------------------------
get_params <- function(format = c("naked", "fab", "fc"),
                       species = c("rabbit", "human")) {
  format <- match.arg(format)
  species <- match.arg(species)

  base <- switch(format,
                 naked = params_naked_rabbit,
                 fab   = params_fab_rabbit,
                 fc    = params_fc_rabbit)

  if (species == "human") {
    base$Vv <- phys_volumes_human$Vv
    base$Va <- phys_volumes_human$Va
    base$Vr <- phys_volumes_human$Vr
    base$Vp <- phys_volumes_human$Vp
    base$k_ao <- k_ao_human
    # Allometric clearance scaling: BW^0.85 (Betts 2018)
    base$CL_p <- base$CL_p * (70/3)^0.85
  }
  unlist(base)
}

cat("Parameter sets defined for naked, fab, fc x rabbit, human.\n")
cat("Use get_params(format, species) to retrieve.\n")
