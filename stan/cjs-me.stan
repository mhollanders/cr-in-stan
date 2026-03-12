functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=S> y;  // detection history
  int<lower=0, upper=1> ind;  // survey (0) or individual-level (1) parameters
  int<lower=0> grainsize;  // threading
}

transformed data {
  int Jm1 = J - 1, Sm1 = S - 1, Sp1 = S + 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int seq = linspaced_int_array(I, 1, I);
  vector[Jm1] tau_scl = tau / exp(mean(log(tau)));
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, Jm1] p;  // detection probabilities
  vector<lower=0, upper=1>[Sm1] delta;  // event probabilities
  simplex[S] eta;  // initial state probabilities
}

transformed parameters {
  // priors
  real lprior = gamma_lpdf(h | 1, 3) + gamma_lpdf(q | 1, 3)
                + beta_lpdf(to_vector(p) | 1, 1);
}

model {
  target += lprior;
  
  // log TPMs, detection logits, event and initial state log probabilities
  matrix[Sp1, Sp1] Q = rate_matrix(h, q);
  array[Jm1] matrix[Sp1, Sp1] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j, :S] = log(matrix_exp(Q * tau_scl[j])[:S]);
    log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
  }
  matrix[S, Jm1] logit_p_j = logit(p);
  array[J] matrix[S, S] log_E_j = 
    rep_array(log(triangular_bidiagonal_stochastic_matrix(delta)), J);
  matrix[S, J] log_eta_j = rep_matrix(log(eta), J);
            
  // likelihood with individual or survey-level parameters
  if (ind) {
    array[I, Jm1] matrix[Sp1, Sp1] log_H_i = rep_array(log_H_j, I);
    array[I] matrix[S, Jm1] logit_p_i = rep_array(logit_p_j, I);
    array[I, J] matrix[S, S] log_E_i = rep_array(log_E_j, I);
    array[I] matrix[S, J] log_eta_i = rep_array(log_eta_j, I);
    target += grainsize ? 
              reduce_sum(partial_cjs_me, seq, grainsize, y, f_l, log_H_i,
                         logit_p_i, log_E_i, log_eta_i)
              : sum(cjs_me(y, f_l, log_H_i, logit_p_i, log_E_i, log_eta_i));
    
  } else {
    target += cjs_me(y, f_l, log_H_j, logit_p_j, log_E_j, log_eta_j);
  }
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[Sp1, Sp1] Q = rate_matrix(h, q);
    array[Jm1] matrix[Sp1, Sp1] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j, :S] = log(matrix_exp(Q * tau_scl[j])[:S]);
      log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    matrix[S, Jm1] logit_p_j = logit(p);
    array[J] matrix[S, S] log_E_j = 
      rep_array(log(triangular_bidiagonal_stochastic_matrix(delta)), J);
    matrix[S, J] log_eta_j = rep_matrix(log(eta), J);
    if (ind) {
      array[I, Jm1] matrix[Sp1, Sp1] log_H_i = rep_array(log_H_j, I);
      array[I] matrix[S, Jm1] logit_p_i = rep_array(logit_p_j, I);
      array[I, J] matrix[S, S] log_E_i = rep_array(log_E_j, I);
      array[I] matrix[S, J] log_eta_i = rep_array(log_eta_j, I);
      log_lik = cjs_me(y, f_l, log_H_i, logit_p_i, log_E_i, log_eta_i);
    } else {
      log_lik = cjs_me(y, f_l, log_H_j, logit_p_j, log_E_j, log_eta_j);
    }
  }
}
