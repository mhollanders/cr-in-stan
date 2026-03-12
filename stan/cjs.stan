functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=1> y;  // detection history
  int<lower=0, upper=1> ind;  // survey (0) or individual-level (1) parameters
  int<lower=0> grainsize;  // threading
}

transformed data {
  int Jm1 = J - 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int seq = linspaced_int_array(I, 1, I);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  vector<lower=0, upper=1>[Jm1] p;  // detection probabilities
}

transformed parameters {
  // priors
  real lprior = gamma_lpdf(h | 1, 3) + beta_lpdf(p | 1, 1);
}

model {
  target += lprior;
  
  // log survival probabilities and detection logits
  vector[Jm1] log_phi_j = -h * tau,
              logit_p_j = logit(p);
  
  // likelihood with individual or survey-level parameters
  if (ind) {
    matrix[Jm1, I] log_phi_i = rep_matrix(log_phi_j, I),
                   logit_p_i = rep_matrix(logit_p_j, I);
    target += grainsize ?
              reduce_sum(partial_cjs, seq, grainsize, y, f_l, log_phi_i, 
                         logit_p_i)
              : sum(cjs(y, f_l, log_phi_i, logit_p_i));
  } else {
    target += cjs(y, f_l, log_phi_j, logit_p_j);
  }
}

generated quantities {
  vector[I] log_lik;
  {
    vector[Jm1] log_phi_j = -h * tau,
                logit_p_j = logit(p);
    if (ind) {
      matrix[Jm1, I] log_phi_i = rep_matrix(log_phi_j, I),
                     logit_p_i = rep_matrix(logit_p_j, I);
      log_lik = cjs(y, f_l, log_phi_i, logit_p_i);
    } else {
      log_lik = cjs(y, f_l, log_phi_j, logit_p_j);
    }
  }
}
