functions {
  #include util.stanfunctions
  #include js.stanfunctions
  #include js-rng.stanfunctions
}

data {
  int<lower=1> I,  // number of individuals
               J,  // number of surveys
               K_max;  // maximum number of secondaries
  array[J] int<lower=1, upper=K_max> K;  // number of secondaries
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J, K_max] int<lower=0, upper=S> y;  // detection history
  int<lower=1> I_aug;  // number of augmented individuals
  int<lower=0, upper=1> dirichlet,  // logistic-normal (0) or Dirichlet (1) entry
                        ind,  // survey (0) or individual-level (1) parameters
                        collapse;  // full (0) or collapsed (1) augmented likelihood
}

transformed data {
  int I_all = I + I_aug, Ip1 = I + 1, Jm1 = J - 1, Sm1 = S - 1, Sp1 = S + 1;
  array[I, 2] int f_l = first_last(y);
  vector[Jm1] tau_scl = tau / exp(mean(log(tau))), log_tau_scl = log(tau_scl);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, J] p;  // detection probabilities
  real<lower=0> mu;  // Dirichlet concentration or logistic-normal scale
  real gamma;  // first entry offset
  sum_to_zero_vector[J] u;  // unconstrained entries
  simplex[S] eta;  // entry state probabilities
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
  real lprior = gamma_lpdf(h | 1, 3) + gamma_lpdf(q | 1, 3)
                + beta_lpdf(to_vector(p) | 1, 1)
                + gamma_lpdf(mu | 1, 1) + std_normal_lpdf(gamma);
}

model {
  target += lprior;
  target += dirichlet ?
            dirichlet_lupdf(exp(log_beta) | exp(log_alpha)) 
            : std_normal_lupdf(u);
            
  // TPMs, detection logits, and initial state log probabilities
  matrix[S, S] Q = rate_matrix(h, q)[:S, :S];
  array[Jm1] matrix[S, S] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j] = log(matrix_exp(Q * tau_scl[j]));
  }
  array[J] matrix[S, K_max] logit_p_j;
  for (j in 1:J) {
    logit_p_j[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
  }
  matrix[S, J] log_eta_j = rep_matrix(log(eta), J);
  
  // likelihood with individual or survey-level parameters
  if (ind) {
    
    // collapsed likelihood (shared for all augmented)
    if (collapse) {
      array[Ip1, Jm1] matrix[S, S] log_H_i = rep_array(log_H_j, Ip1);
      array[Ip1, J] matrix[S, K_max] logit_p_i = rep_array(logit_p_j, Ip1);
      array[Ip1] matrix[S, J] log_eta_i = rep_array(log_eta_j, Ip1);
      tuple(vector[I], vector[2], matrix[J, I], vector[J], 
            array[I] matrix[S, J], matrix[S, J]) lp = 
        js_ms_rd2(y, f_l, K, log_H_i, logit_p_i, log_beta, log_eta_i, psi);
      target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
      
      // full likelihood (unique for all augmented)
     } else {
      array[I_all, Jm1] matrix[S, S] log_H_i = rep_array(log_H_j, I_all);
      array[I_all, J] matrix[S, K_max] logit_p_i = rep_array(logit_p_j, I_all);
      array[I_all] matrix[S, J] log_eta_i = rep_array(log_eta_j, I_all);
      tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug], 
            array[I] matrix[S, J], array[I_aug] matrix[S, J]) lp = 
        js_ms_rd(y, f_l, K, log_H_i, logit_p_i, log_beta, log_eta_i, psi);
      target += lp.1;
      for (i in 1:I_aug) {
        target += log_sum_exp(lp.2[:, i]);
      }
    }
    
    // likelihood with survey varying parameters
  } else {
    tuple(vector[I], vector[2], matrix[J, I], vector[J], array[I] matrix[S, J],
          matrix[S, J]) lp =
      js_ms_rd(y, f_l, K, log_H_j, logit_p_j, log_beta, log_eta_j, psi);
    target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
  }
}

generated quantities {
  vector[I] log_lik;
  array[S, J] int N, B, D;
  int N_super;
  {
    matrix[Sp1, Sp1] Q = rate_matrix(h, q);
    array[Jm1] matrix[Sp1, Sp1] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j] = log(matrix_exp(Q * tau_scl[j]));
    }
    array[J] matrix[S, K_max] logit_p_j;
    for (j in 1:J) {
      logit_p_j[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
    }
    matrix[S, J] log_eta_j = rep_matrix(log(eta), J);
    tuple(array[S, J] int, array[S, J] int, array[S, J] int, int) latent;
    if (ind) {
      if (collapse) {
        array[Ip1, Jm1] matrix[Sp1, Sp1] log_H_i = rep_array(log_H_j, Ip1);
        array[Ip1, J] matrix[S, K_max] logit_p_i = rep_array(logit_p_j, Ip1);
        array[Ip1] matrix[S, J] log_eta_i = rep_array(log_eta_j, Ip1);
        tuple(vector[I], vector[2], matrix[J, I], vector[J], 
              array[I] matrix[S, J], matrix[S, J]) lp = 
          js_ms_rd2(y, f_l, K, log_H_i[:, :, :S, :S], logit_p_i, log_beta, 
                    log_eta_i, psi);
        log_lik = lp.1;
        latent = js_ms_rd2_rng(lp, y, f_l, K, log_H_i, logit_p_i, I_aug);
      } else {
        array[I_all, Jm1] matrix[Sp1, Sp1] log_H_i = rep_array(log_H_j, I_all);
        array[I_all, J] matrix[S, K_max] logit_p_i = rep_array(logit_p_j, I_all);
        array[I_all] matrix[S, J] log_eta_i = rep_array(log_eta_j, I_all);
        tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug], 
              array[I] matrix[S, J], array[I_aug] matrix[S, J]) lp = 
          js_ms_rd(y, f_l, K, log_H_i[:, :, :S, :S], logit_p_i, log_beta, 
                   log_eta_i, psi);
        log_lik = lp.1;
        latent = js_ms_rd_rng(lp, y, f_l, K, log_H_i, logit_p_i, I_aug);
      }
    } else {
      tuple(vector[I], vector[2], matrix[J, I], vector[J], 
            array[I] matrix[S, J], matrix[S, J]) lp =
        js_ms_rd(y, f_l, K, log_H_j[:, :S, :S], logit_p_j, log_beta, log_eta_j, 
                 psi);
      log_lik = lp.1;
      latent = js_ms_rd_rng(lp, y, f_l, K, log_H_j, logit_p_j, I_aug);
    }
    N = latent.1;
    B = latent.2;
    D = latent.3;
    N_super = latent.4;
  }
}
