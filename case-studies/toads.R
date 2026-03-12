# load packages
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, cmdstanr, loo, tidybayes, ggh4x, MetBrewer)
source(here("sbc/util.R"))
cores <- 8
options(mc.cores = cores)
theme_set(my_theme())

# prepare data
toads <- read_csv(here("case-studies/data/toads.csv"))
dh <- as.matrix(toads[, -(1:3)])
dates <- mdy(colnames(dh))
J <- length(dates)
stan_data <- list(I = nrow(dh),
                  J = J,
                  tau = as.numeric(diff(dates, units = "weeks")), 
                  y = dh, 
                  I_aug = 300) |> 
  glimpse()

# fit and compare
mod <- cmdstan_model(here("case-studies/stan/js-toads.stan"))
fits <- map(0:2, ~{
  stan_data$config <- .
  mod$sample(stan_data, refresh = 0, chains = cores,
             iter_warmup = 500, iter_sampling = 500, show_exceptions = F,
             init = mod$pathfinder(stan_data, init = 0.1, sig_figs = 14,
                                   num_paths = cores, single_path_draws = 100,
                                   max_lbfgs_iters = 200, psis_resample = F))
})
loos <- map(fits, ~.$loo())
loo_compare(loos)

# plot
map(fits[c(1, 3)], ~gather_rvars(., beta[j], N[j])) |> 
  list_rbind(names_to = "mod") |> 
  ggplot(aes(factor(j, labels = dates) |> ymd(), 
             ydist = .value)) +
  facet_wrap(~ factor(.variable, labels = c("Entry~Probabilities~(beta)",
                                            "Population~Size~(N)")), 
             ncol = 1, scales = "free_y", 
             labeller = label_parsed) +
  stat_pointinterval(aes(colour = factor(mod)),
                     point_interval = median_hdci, .width = 0.95,
                     size = 0.1, linewidth = 0.1,
                     position = position_dodge(width = 1)) + 
  scale_x_date(date_labels = "%b %y", 
               date_breaks = "3 week") + 
  scale_colour_manual(values = c("#dd5129", "#0f7ba2"),
                      labels = scales::label_parse()(c(
                        "bold(beta) %~% Dirichlet(bold(1))",
                        "bold(beta) %~% Dirichlet(mu * bold(tau))"
                      ))) +
  facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(0.2, 0.6, 0.2),
                                                  limits = c(0, 0.7),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(50, 150, 50),
                                                  limits = c(0, 175),
                                                  expand = c(0, 0)))) +
  theme(legend.position = "inside",
        legend.position.inside = c(0.95, 0.95),
        legend.justification = c("right", "top")) + 
  labs(x = "Survey", y = "Estimate (95% HDI)", colour = "Prior")
ggsave(here("manuscript/figs/fig-toad.jpg"), width = 8, height = 7, dpi = 600)
