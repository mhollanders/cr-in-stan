if (!require(expm)) install.packages("expm")
simulate_cmr <- function(N_super = 100, J = 8, K_max = 1, S = 1, I_aug = 500,
                         ME = FALSE, JS = FALSE, dirichlet = FALSE, 
                         ind = FALSE, collapse = FALSE, grainsize = 0,
                         mu_gamma_prior = c(1, 1), gamma_normal_prior = c(0, 1),
                         h_gamma_prior = c(1, 3), q_gamma_prior = c(1, 3), 
                         p_beta_prior = c(1, 1), eta_dirichlet_prior = 1, 
                         delta_beta_prior = c(1, 1)) {
  
  # transformed data and parameters
  Jm1 <- J - 1
  if (K_max > 1) {
    K <- sample(1:K_max, J, replace = T, prob = sort(runif(K_max)))
    while (K[1] == 1) {
      K[1] <- sample(2:K_max, 1)
    }
    K_max <- max(K)
  } else {
    K <- rep(1, J)
  }
  tau <- rlnorm(Jm1)
  tau_scl <- tau / exp(mean(log(tau)))
  h <- rgamma(S, h_gamma_prior[1], h_gamma_prior[2])
  p <- matrix(rbeta(S * J, p_beta_prior[1], p_beta_prior[2]), S, J)
  if (S == 1) {
    phi_tau <- exp(-h * tau)
  } else {
    Sm1 <- S - 1 ; Sp1 <- S + 1
    q <- rgamma(S * Sm1, q_gamma_prior[1], q_gamma_prior[2])
    H <- array(0, c(Jm1, Sp1, Sp1))
    for (j in 1:Jm1) {
      H[j, , ] <- expm::expm(rate_matrix(h, q) * tau_scl[j])
    }
    if (ME) {
      delta <- rbeta(Sm1, delta_beta_prior[1], delta_beta_prior[2])
      E <- triangular_bidiagonal_stochastic_matrix(delta)
    }
    if (ME || JS) {
      if (length(eta_dirichlet_prior) == 1) {
        alpha <- rep(eta_dirichlet_prior, S)
      } else {
        alpha <- eta_dirichlet_prior
      }
      eta <- rdirch(1, alpha)
    }
  }
  
  # latent states and detection history
  z <- matrix(0, N_super, J)
  y <- array(0, c(N_super, J, K_max))
  
  # Jolly-Seber
  if (JS) {
    mu <- rgamma(1, mu_gamma_prior[1], mu_gamma_prior[2])
    gamma <- rnorm(1, gamma_normal_prior[1], gamma_normal_prior[2])
    log_alpha <- c(gamma, log(tau_scl))
    if (dirichlet) {
      beta <- rdirch(1, exp(log(mu) + log_alpha))
    } else {
      u <- rnorm(J, log_alpha, mu)
      beta <- softmax(u - mean(u))
    }
    b <- sort(rcat(N_super, beta))
    
    # single state ecological process
    if (S == 1) {
      B <- D <- numeric(J)
      for (i in 1:N_super) {
        b_i <- b[i]
        z[i, b_i] <- 1
        B[b_i] <- B[b_i] + 1
        if (b_i < J) {
          for (j in (b_i + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rbinom(1, 1, z[i, jm1] * phi_tau[jm1])
            if (z[i, jm1] == 1 & z[i, j] == 0) {
              D[j] <- D[j] + 1
            }
          }
        }
        
        # observation process
        for (j in b_i:J) {
          y[i, j, 1:K[j]] <- rbinom(K[j], 1, z[i, j] * p[j])
        }
      }
      
      # multistate ecological process
    } else {
      B <- D <- matrix(0, S, J)
      for (i in 1:N_super) {
        b_i <- b[i]
        z[i, b_i] <- rcat(1, eta)
        B[z[i, b_i], b_i] <- B[z[i, b_i], b_i] + 1
        if (b_i < J) {
          for (j in (b_i + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rcat(1, H[jm1, z[i, jm1], ])
            if (z[i, jm1] < Sp1 & z[i, j] == Sp1) {
              D[z[i, jm1], j] <- D[z[i, jm1], j] + 1
            }
          }
        }
        
        # observation process
        for (j in b_i:J) {
          if (z[i, j] < Sp1) {
            y[i, j, 1:K[j]] <- rbinom(K[j], 1, p[z[i, j], j]) * z[i, j]
            
            # multievent
            if (ME) {
              for (k in 1:K[j]) {
                if (y[i, j, k]) {
                  y[i, j, k] <- rcat(1, E[z[i, j], ])
                }
              }
            }
          }
        }
      }
    }
    
    # subset observed individuals and secondaries
    obs <- which(rowSums(y) > 0)
    I <- length(obs)
    y <- y[obs, , 1:K_max]
    
    # single state CJS 
  } else {
    I <- N_super
    f <- sort(sample(1:ifelse(K_max > 1 || ME, J, Jm1), I, replace = T))
    if (S == 1) {
      for (i in 1:I) {
        f_i <- f[i]
        g <- sample(1:K[f_i], 1)
        z[i, f_i] <- y[i, f_i, g] <- 1
        
        # observation process conditioned on first capture
        for (k in setdiff(1:K[f_i], g)) {
          y[i, f_i, k] <- rbinom(1, 1, p[f_i])
        }
        if (f_i < J) {
          for (j in (f_i + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rbinom(1, 1, z[i, jm1] * phi_tau[jm1])
            y[i, j, 1:K[j]] <- rbinom(1:K[j], 1, z[i, j] * p[j])
          }
        }
      }
      
      # multistate/multievent
    } else {
      for (i in 1:I) {
        f_i <- f[i]
        g <- sample(1:K[f_i], 1)
        if (ME) {
          z[i, f_i] <- rcat(1, eta)
          y[i, f_i, g] <- rcat(1, E[z[i, f_i], ])
        } else {
          z[i, f_i] <- y[i, f_i, g] <- sample(1:S, 1)
        }
        for (k in setdiff(1:K[f_i], g)) {
          y[i, f_i, k] <- rbinom(1, 1, p[z[i, f_i], f_i]) * z[i, f_i]
          if (ME) {
            if (y[i, f_i, k]) {
              y[i, f_i, k] <- rcat(1, E[z[i, f_i], ])
            }
          }
        }
        if (f_i < J) {
          for (j in (f_i + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rcat(1, H[jm1, z[i, jm1], ])
            if (z[i, j] < Sp1) {
              y[i, j, 1:K[j]] <- rbinom(K[j], 1, p[z[i, j], j]) * z[i, j]
              if (ME) {
                for (k in 1:K[j]) {
                  if (y[i, j, k]) {
                    y[i, j, k] <- rcat(1, E[z[i, j], ])
                  }
                }
              }
            }
          }
        }
      }
    }
    
    # reduce dimensions if single survey
    y <- y[, , 1:K_max]
  }
  
  
  # modify dimensions
  if (JS || K_max > 1) {
    p <- p[1:S, ]
  } else {
    p <- p[1:S, -1]
  }
  
  # initiate output as CJS
  variables <- list(h = h, p = p)
  generated <- list(I = I, J = J, tau = tau, y = y, ind = ind)
  
  # update robust design
  if (K_max > 1) {
    generated <- append(generated, list(K_max = K_max, K = K))
  }
  
  # update multistate/multievent
  if (S > 1) {
    generated$S <- S
    variables$q <- q
    if (ME) {
      if (S == 2) {
        variables$`delta[1]` <- delta
      } else {
        variables$delta <- delta
      }
    }
    if (ME || JS) {
      variables$eta <- eta
    }
  }
  
  # update Jolly-Seber
  if (JS) {
    generated <- append(generated, list(I_aug = I_aug, dirichlet = dirichlet, 
                                        collapse = collapse))
    variables <- append(variables,
                        list(mu = mu,
                             gamma = gamma,
                             log_beta = log(beta),
                             N_super = N_super,
                             B = B,
                             D = D,
                             N = apply(z, 2, \(j) 
                                       sapply(1:S, \(s) sum(j == s)))))
    
    # grainsize for CJS within-chain parallelisation
  } else {
    generated$grainsize <- grainsize
  }
  
  list(variables = variables, generated = generated)
}
