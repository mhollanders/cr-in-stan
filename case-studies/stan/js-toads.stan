functions {
  #include ../../stan/util.stanfunctions
  #include ../../stan/js.stanfunctions
  #include ../../stan/js-rng.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=1> y;  // detection history
  int<lower=1> I_aug;  // number of augmented individuals
  int<lower=0, upper=2> config;  // entry configuration
}

transformed data {
  int Jm1 = J - 1;
  array[I, 2] int f_l = first_last(y);
  vector[Jm1] tau_scl = tau / exp(mean(log(tau))), log_tau_scl = log(tau_scl);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  real<lower=0, upper=1> p;  // detection probability
  vector<lower=0>[config > 0] mu;  // Dirichlet concentration or logistic-normal scale
  vector<lower=0>[config == 2] gamma;  // first entry offset
  simplex[J] beta;  // entry probabilities
  real<lower=0, upper=1> psi;  // inclusion probability
}

transformed parameters {
  // entry concentrations
  vector[J] log_alpha;
  if (config == 0) {
    log_alpha = zeros_vector(J);  // uniform
  } else {
    log_alpha = rep_vector(log(mu[1]), J);  // concentration estimated
    if (config == 2) {
      log_alpha += append_row(log(gamma[1]), log_tau_scl);  // intervals accomodated
    }
  }
  
  // priors
  real lprior = gamma_lpdf(h | 1, 3) + beta_lpdf(p | 1, 1) 
                + gamma_lpdf(mu | 1, 1) + gamma_lpdf(gamma | 1, 1)
                + dirichlet_lpdf(beta | exp(log_alpha));
}

model {
  target += lprior;
  
  // likelihood
  vector[Jm1] log_phi = -h * tau;
  vector[J] logit_p = rep_vector(logit(p), J);
  tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp =
    js(y, f_l, log_phi, logit_p, log(beta), psi);
  target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
}

generated quantities {
  vector[I] log_lik;
  array[J] int N, B, D;
  int N_super;
  {
    vector[Jm1] log_phi = -h * tau;
    vector[J] logit_p = rep_vector(logit(p), J);
    tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp =
      js(y, f_l, log_phi, logit_p, log(beta), psi);
    log_lik = lp.1;
    tuple(array[J] int, array[J] int, array[J] int, int) latent =
      js_rng(lp, f_l, log_phi, logit_p, I_aug);
    N = latent.1;
    B = latent.2;
    D = latent.3;
    N_super = latent.4;
  }
}
