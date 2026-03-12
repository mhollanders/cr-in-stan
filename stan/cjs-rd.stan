functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I,  // number of individuals
               J,  // number of surveys
               K_max;  // maximum number of secondaries
  array[J] int<lower=1, upper=K_max> K;  // number of secondaries
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J, K_max] int<lower=0, upper=1> y;  // detection history
  int<lower=0, upper=1> ind;  // survey (0) or individual-level (1) parameters
  int<lower=0> grainsize;  // threading
}

transformed data {
  int Jm1 = J - 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int g = first_sec(y, f_l[:, 1]);
  array[I] int seq = linspaced_int_array(I, 1, I);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  row_vector<lower=0, upper=1>[J] p;  // detection probabilities
}

transformed parameters {
  // priors
  real lprior = gamma_lpdf(h | 1, 3) + beta_lpdf(p | 1, 1);
}

model {
  target += lprior;
  
  // log survival probabilities and detection logits
  vector[Jm1] log_phi_j = -h * tau;
  matrix[K_max, J] logit_p_j;
  for (j in 1:J) {
    logit_p_j[:K[j], j] = rep_vector(logit(p[j]), K[j]);
  }
  
  // likelihood with individual or survey-level parameters
  if (ind) {
    matrix[Jm1, I] log_phi_i = rep_matrix(log_phi_j, I);
    array[I] matrix[K_max, J] logit_p_i = rep_array(logit_p_j, I);
    target += grainsize ?
              reduce_sum(partial_cjs_rd, seq, grainsize, y, f_l, K, g,
                         log_phi_i, logit_p_i)
              : sum(cjs_rd(y, f_l, K, g, log_phi_i, logit_p_i));
  } else {
    target += cjs_rd(y, f_l, K, g, log_phi_j, logit_p_j);
  }
}

generated quantities {
  vector[I] log_lik;
  {
    vector[Jm1] log_phi_j = -h * tau;
    matrix[K_max, J] logit_p_j;
    for (j in 1:J) {
      logit_p_j[:K[j], j] = rep_vector(logit(p[j]), K[j]);
    }
    if (ind) {
      matrix[Jm1, I] log_phi_i = rep_matrix(log_phi_j, I);
      array[I] matrix[K_max, J] logit_p_i = rep_array(logit_p_j, I);
      log_lik = cjs_rd(y, f_l, K, g, log_phi_i, logit_p_i);
    } else {
      log_lik = cjs_rd(y, f_l, K, g, log_phi_j, logit_p_j);
    }
  }
}
