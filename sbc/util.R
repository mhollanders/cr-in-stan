# ggplot theme
my_theme <- function(base_size = 8,
                     base_family = "", 
                     base_line_size = base_size / 20, 
                     base_rect_size = base_size / 20) {
  # Set color and half-line size for margins
  my_black <- "#333333"
  half_line <- base_size / 2
  
  # Base theme customization
  theme_grey(base_size = base_size,
             base_family = base_family, 
             base_line_size = base_line_size,
             base_rect_size = base_rect_size) %+replace%
    theme(
      # Axis settings
      axis.line = element_blank(),
      axis.text = element_text(colour = my_black, size = rel(0.9)),
      axis.title = element_text(colour = my_black, size = rel(1)),
      axis.ticks = element_line(colour = my_black),
      
      # Legend
      legend.key = element_rect(fill = "white", colour = NA),
      legend.text = element_text(size = rel(0.9)),
      
      # Panel
      panel.background = element_rect(fill = NA, colour = NA),
      panel.border = element_rect(fill = NA, colour = my_black),
      panel.grid = element_blank(),
      
      # Plot settings
      plot.margin = margin(10, 10, 10, 10),
      plot.title = element_text(
        size = rel(1.1), 
        hjust = 0, 
        vjust = 1, 
        margin = margin(b = half_line)
      ),
      
      # Strip settings (e.g., facet labels)
      strip.background = element_rect(
        fill = my_black, 
        colour = my_black, 
        linewidth = base_line_size / 2
      ),
      strip.text = element_text(
        colour = "white", 
        size = rel(0.9), 
        margin = margin(0.8 * half_line)
      ),
      
      # Text color
      text = element_text(colour = my_black),
      
      # Mark as complete theme
      complete = TRUE
    )
}

# get event probabilities from lower triangular event matrix
delta_from_E <- function(E) {
  S <- nrow(E)
  delta <- numeric(choose(S, 2) + S - 1)
  idx <- 1
  for (s in 2:S) {
    sm1 <- s - 1
    delta[idx:(idx + sm1)] <- E[s, 1:s]
    idx <- idx + s
  }
  delta
}

# lower triangular row stochastic matrix from vector of probabilities
triangular_row_stochastic_matrix <- function(S, u) {
  E <- diag(S)
  idx <- 1
  for (s in 2:S) {
    sm1 <- s - 1
    u_s <- u[idx:(idx + sm1)]
    E[s, 1:s] <- u_s / sum(u_s)
    idx <- idx + s
  }
  E
}

# transition rate matrix from vectors of mortality and transition rates
rate_matrix <- function(h, q) {
  S <- length(h)
  Sp1 <- S + 1 ; Sm1 <- S - 1
  Q <- matrix(0, Sp1, Sp1)
  q_s <- head(q, Sm1)
  Q[1, 1] <- -(h[1] + sum(q_s))
  Q[1, 2:S] <- q_s
  if (S > 2) {
    idx <- S
    for (s in 2:Sm1) {
      q_s <- q[idx:(idx + S - 2)]
      Q[s, 1:(s - 1)] <- head(q_s, s - 1)
      Q[s, s] <- -(h[s] + sum(q_s))
      Q[s, (s + 1):S] <- tail(q_s, S - s)
      idx <- idx + Sm1
    }
  }
  q_s <- tail(q, Sm1)
  Q[S, 1:Sm1] <- q_s
  Q[S, S] <- -(h[S] + sum(q_s))
  Q[1:S, Sp1] <- h
  Q
}

# triangular biadiagonal stochastic matrix
triangular_bidiagonal_stochastic_matrix <- function(delta) {
  E <- diag(c(1, delta))
  for (s in 2:(length(delta) + 1)) {
    sm1 <- s - 1
    E[s, sm1:s] <- c(1 - delta[sm1], delta[sm1])
  }
  E
}

# random categorical draws
rcat <- function(n = 1, prob) {
  rmultinom(n, size = 1, prob) |>
    apply(2, \(x) which(x == 1))
}

# random Dirichlet draws
rdirch <- function(n = 1, alpha) {
  D <- length(alpha)
  out <- matrix(NA, n, D)
  for (i in 1:n) {
    u <- rgamma(D, alpha, 1)
    out[i, ] <- u / sum(u)
  }
  if (n == 1) {
    out[1, ]
  } else {
    out
  }
}

# softmax
softmax <- function(x) {
  exp_x <- exp(x)
  exp_x / sum(exp_x)
}

# plot ECDF-diff and estimates plots together
if (!require(patchwork)) install.packages("patchwork")
plot_sbc <- function(sbc, ..., nrow = NULL, ncol = NULL) {
  patchwork::wrap_plots(
    plot_ecdf_diff(sbc, ...) +
      ggplot2::facet_wrap(~ group, 
                          nrow = nrow, 
                          ncol = ncol) + 
      ggplot2::theme(legend.position = "none"), 
    plot_sim_estimated(sbc, ...) +
      ggplot2::facet_wrap(~ variable, 
                          nrow = nrow, 
                          ncol = ncol, 
                          scales = "free"),
    ncol = 1
  )
}
