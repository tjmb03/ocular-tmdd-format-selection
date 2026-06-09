# ============================================================================
# Ocular TMDD Model Definition (rxode2)
# ----------------------------------------------------------------------------
# Four-compartment intravitreal PK-TMDD model for three molecular formats
# of a Microglial Target Ligand (MTL):
#   - Naked peptide (~30 kDa, monovalent, no FcRn)
#   - Fab-fusion   (~80 kDa, monovalent, no FcRn)
#   - Fc-fusion    (~115 kDa, bivalent, FcRn-engaged)
#
# Compartments:
#   1. Vitreous          (Cv) - intravitreal dosing site
#   2. Aqueous humor     (Ca) - anterior elimination route (DOMINANT, ~75%)
#   3. Retina/microglia  (Cr, R, DR) - posterior route, TMDD + FcRn here
#   4. Systemic plasma   (Cp) - escape compartment
#
# Format-specific biology lives entirely in parameter values, not structure.
# See docs/methods.md for full derivation.
# ============================================================================

library(rxode2)

ocular_tmdd_model <- rxode2({

  # --- Concentrations (nM = nmol / mL) ---
  Cv = A_Cv / Vv
  Ca = A_Ca / Va
  Cr = A_Cr / Vr
  R  = A_R  / Vr
  DR = A_DR / Vr
  Cp = A_Cp / Vp

  # --- Binding flux (mass-action, in retina compartment) ---
  bind_flux   = k_on  * Cr * R          # nM/h
  unbind_flux = k_off * DR              # nM/h

  # --- FcRn recycling flux (only Fc-fusion: phi_FcRn = 1) ---
  recycle_flux = phi_FcRn * f_rec * k_int * DR     # nM/h

  # --- Vitreous: free drug only, no target ---
  d/dt(A_Cv) = -k_va * A_Cv + k_av * A_Ca - k_vr * A_Cv + k_rv * A_Cr

  # --- Aqueous humor: receives from vitreous, exits via trabecular outflow ---
  d/dt(A_Ca) = k_va * A_Cv - k_av * A_Ca - k_ao * A_Ca

  # --- Retina free drug: TMDD active here ---
  d/dt(A_Cr) = k_vr * A_Cv - k_rv * A_Cr -
               (bind_flux - unbind_flux) * Vr +
               recycle_flux * Vr

  # --- Free target on microglia ---
  d/dt(A_R)  = (k_syn - k_deg * R) * Vr -
               (bind_flux - unbind_flux) * Vr

  # --- Drug-target complex ---
  d/dt(A_DR) = (bind_flux - unbind_flux) * Vr -
               k_int * A_DR

  # --- Systemic plasma: anterior outflow + first-order elimination ---
  d/dt(A_Cp) = k_ao * A_Ca - (CL_p / Vp) * A_Cp

  # --- Derived readouts for plotting/analysis ---
  TO       = DR / (R + DR + 1e-12)               # target occupancy (0-1)
  C_total  = Cr + DR                              # total drug in retina

})

cat("Model compiled successfully.\n")
cat("State variables:", paste(ocular_tmdd_model$state, collapse = ", "), "\n")
cat("Parameters:",     paste(ocular_tmdd_model$params, collapse = ", "), "\n")
