functions {
  #include util.stanfunctions
  #include js.stanfunctions
  #include js-rng.stanfunctions
}

data {
  int<lower=1> I,  // number of individuals
               J,  // number of surveys
               K_max;  // maximum number of secondaries
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[J] int<lower=0, upper=K_max> K;  // number of secondaries
  array[I, J, K_max] int<lower=0, upper=1> y;  // detection history
  int<lower=1> I_aug;  // number of augmented individuals
  int<lower=0, upper=1> dirichlet,  // logistic-normal (0) or Dirichlet (1) entry
                        ind,  // survey (0) or individual-level (1) parameters
                        collapse;  // full (0) or collapsed (1) augmented likelihood
}

transformed data {
  int I_all = I + I_aug, Ip1 = I + 1, Jm1 = J - 1;
  array[I, 2] int f_l = first_last(y);
  vector[Jm1] tau_scl = tau / exp(mean(log(tau))), log_tau_scl = log(tau_scl);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  row_vector<lower=0, upper=1>[J] p;  // detection probabilities
  real<lower=0> mu;  // Dirichlet concentration or logistic-normal scale
  real gamma;  // first entry offset
  sum_to_zero_vector[J] u;  // unconstrained entries
  real<lower=0, upper=1> psi;  // inclusion probability
}

transformed parameters {
  // entry probabilities
  vector[J] log_alpha = append_row(gamma, log_tau_scl), 
            log_beta;
  if (dirichlet) {
    log_alpha += log(mu);
    log_beta = sum_to_zero_log_simplex_jacobian(u);
  } else {
    log_alpha += mu * u;
    log_beta = log_softmax(log_alpha);
  }
  
  // priors
  real lprior = gamma_lpdf(h | 1, 3) + beta_lpdf(p | 1, 1) 
                + gamma_lpdf(mu | 1, 1) + std_normal_lpdf(gamma);
}

model {
  target += lprior;
  target += dirichlet ? 
            dirichlet_lupdf(exp(log_beta) | exp(log_alpha)) 
            : std_normal_lupdf(u);
            
  // log survival probabilities and detection logits
  vector[Jm1] log_phi_j = -h * tau;
  matrix[K_max, J] logit_p_j;
  for (j in 1:J) {
    logit_p_j[:K[j], j] = rep_vector(logit(p[j]), K[j]);
  }
            
  // likelihood with individual-by-survey varying parameters
  if (ind) {
    
    // collapsed likelihood (shared for all augmented)
    if (collapse) {
      matrix[Jm1, Ip1] log_phi_i = rep_matrix(log_phi_j, Ip1);
      array[Ip1] matrix[K_max, J] logit_p_i = rep_array(logit_p_j, Ip1);
      tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp =
        js_rd2(y, f_l, K, log_phi_i, logit_p_i, log_beta, psi);
      target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
      
      // full likelihood (unique for all augmented)
    } else {
      matrix[Jm1, I_all] log_phi_i = rep_matrix(log_phi_j, I_all);
      array[I_all] matrix[K_max, J] logit_p_i = rep_array(logit_p_j, I_all);
      tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug]) lp =
            js_rd(y, f_l, K, log_phi_i, logit_p_i, log_beta, psi);
      target += lp.1;
      for (i in 1:I_aug) {
        target += log_sum_exp(lp.2[:, i]);
      }
    }
    
    // likelihood with survey varying parameters
  } else {
    tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp =
      js_rd(y, f_l, K, log_phi_j, logit_p_j, log_beta, psi);
    target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
  }
}

generated quantities {
  vector[I] log_lik;
  array[J] int N, B, D;
  int N_super;
  {
    vector[Jm1] log_phi_j = -h * tau;
    matrix[K_max, J] logit_p_j;
    for (j in 1:J) {
      logit_p_j[:K[j], j] = rep_vector(logit(p[j]), K[j]);
    }
    tuple(array[J] int, array[J] int, array[J] int, int) latent;
    if (ind) {
      if (collapse) {
        matrix[Jm1, Ip1] log_phi_i = rep_matrix(log_phi_j, Ip1);
        array[Ip1] matrix[K_max, J] logit_p_i = rep_array(logit_p_j, Ip1);
        tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp =
          js_rd2(y, f_l, K, log_phi_i, logit_p_i, log_beta, psi);
        log_lik = lp.1;
        latent = js_rd2_rng(lp, f_l, K, log_phi_i, logit_p_i, I_aug);
      } else {
        matrix[Jm1, I_all] log_phi_i = rep_matrix(log_phi_j, I_all);
        array[I_all] matrix[K_max, J] logit_p_i = rep_array(logit_p_j, I_all);
        tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug]) lp =
          js_rd(y, f_l, K, log_phi_i, logit_p_i, log_beta, psi);
        log_lik = lp.1;
        latent = js_rd_rng(lp, f_l, K, log_phi_i, logit_p_i, I_aug);
      }
    } else {
      tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp =
        js_rd(y, f_l, K, log_phi_j, logit_p_j, log_beta, psi);
      log_lik = lp.1;
      latent = js_rd_rng(lp, f_l, K, log_phi_j, logit_p_j, I_aug);
    }
    N = latent.1;
    B = latent.2;
    D = latent.3;
    N_super = latent.4;
  }
}
