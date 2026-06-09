# ============================================================================
# Forward Simulation of All Three Formats (rxode2)
# ----------------------------------------------------------------------------
# Runs the deterministic 4-compartment model for naked, Fab-fusion, and
# Fc-fusion at a typical 2 mg intravitreal dose in rabbit.
#
# Outputs:
#   - Vitreous concentration-time profiles
#   - Retinal exposure
#   - Target occupancy
#   - Plasma spillover
#
# Run prerequisite: 01_model_definition.R, 02_parameters.R
# ============================================================================

library(rxode2)
library(dplyr)
library(ggplot2)
library(tidyr)

source("model/01_model_definition.R")
source("model/02_parameters.R")

# ----------------------------------------------------------------------------
# Dose specification
# ----------------------------------------------------------------------------
mw_kDa <- list(naked = 30, fab = 80, fc = 115)
dose_mg <- 2.0
dose_nmol <- function(fmt) dose_mg * 1e6 / (mw_kDa[[fmt]] * 1000)

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
sim_horizon_h <- 60 * 24
times <- seq(0, sim_horizon_h, by = 0.5)

run_format <- function(fmt) {
  p <- get_params(fmt, "rabbit")
  inits <- c(A_Cv = 0, A_Ca = 0, A_Cr = 0,
             A_R = target_biology$R0_nM * p["Vr"], A_DR = 0, A_Cp = 0)
  ev <- et(amt = dose_nmol(fmt), cmt = "A_Cv", time = 0) %>% et(times)
  rxSolve(ocular_tmdd_model, params = p, events = ev, inits = inits) %>%
    as.data.frame() %>% mutate(format = fmt)
}

sim_all <- bind_rows(run_format("naked"), run_format("fab"), run_format("fc")) %>%
  mutate(time_d = time / 24,
         format = factor(format, levels = c("naked", "fab", "fc"),
                         labels = c("Naked peptide (30 kDa)",
                                    "Fab-fusion (80 kDa)",
                                    "Fc-fusion (115 kDa)")))

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
fmt_colors <- c("Naked peptide (30 kDa)" = "#E24B4A",
                "Fab-fusion (80 kDa)"    = "#EF9F27",
                "Fc-fusion (115 kDa)"    = "#1D9E75")

theme_qsp <- theme_minimal(base_size = 11) +
  theme(legend.position = "top", legend.title = element_blank())

p_vit <- ggplot(sim_all, aes(time_d, Cv, color = format)) +
  geom_line(linewidth = 0.9) +
  scale_y_log10() + scale_color_manual(values = fmt_colors) +
  labs(x = "Time (days)", y = "Vitreous concentration (nM)",
       title = "Vitreous PK after 2 mg IVT dose") + theme_qsp

ggsave("results/plots/r_vitreous_pk.png", p_vit,
       width = 7, height = 4.5, dpi = 150)

cat("R simulation complete. Plot saved to results/plots/r_vitreous_pk.png\n")
