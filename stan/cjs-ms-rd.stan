functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I,  // number of individuals
               J,  // number of primaries
               K_max;  // maximum number of secondaries
  array[J] int<lower=1, upper=K_max> K;  // number of secondaries
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J, K_max] int<lower=0, upper=S> y;  // detection history
  int<lower=0, upper=1> ind;  // survey (0) or individual-level (1) parameters
  int<lower=0> grainsize;  // threading
}

transformed data {
  int Jm1 = J - 1, Sm1 = S - 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int g = first_sec(y, f_l[:, 1]);
  array[I] int seq = linspaced_int_array(I, 1, I);
  vector[Jm1] tau_scl = tau / exp(mean(log(tau)));
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, J] p;  // detection probabilities
}

transformed parameters {
  // priors
  real lprior = gamma_lpdf(h | 1, 3) + gamma_lpdf(q | 1, 3)
                + beta_lpdf(to_vector(p) | 1, 1);
}

model {
  target += lprior;
  
  // log TPMs and detection logits
  matrix[S, S] Q = rate_matrix(h, q)[:S, :S];
  array[Jm1] matrix[S, S] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j] = log(matrix_exp(Q * tau_scl[j]));
  }
  array[J] matrix[S, K_max] logit_p_j;
  for (j in 1:J) {
    logit_p_j[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
  }
            
  // likelihood with individual or survey-level parameters
  if (ind) {
    array[I, Jm1] matrix[S, S] log_H_i = rep_array(log_H_j, I);
    array[I, J] matrix[S, K_max] logit_p_i = rep_array(logit_p_j, I);
    target += grainsize ?
              reduce_sum(partial_cjs_ms_rd, seq, grainsize, y, f_l, K, g,
                         log_H_i, logit_p_i)
              : sum(cjs_ms_rd(y, f_l, K, g, log_H_i, logit_p_i));
  } else {
    target += cjs_ms_rd(y, f_l, K, g, log_H_j, logit_p_j);
  }
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[S, S] Q = rate_matrix(h, q)[:S, :S];
    array[Jm1] matrix[S, S] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j] = log(matrix_exp(Q * tau_scl[j]));
    }
    array[J] matrix[S, K_max] logit_p_j;
    for (j in 1:J) {
      logit_p_j[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
    }
    if (ind) {
      array[I, Jm1] matrix[S, S] log_H_i = rep_array(log_H_j, I);
      array[I, J] matrix[S, K_max] logit_p_i = rep_array(logit_p_j, I);
      log_lik = cjs_ms_rd(y, f_l, K, g, log_H_i, logit_p_i);
    } else {
      log_lik = cjs_ms_rd(y, f_l, K, g, log_H_j, logit_p_j);
  }
    
  }
}
