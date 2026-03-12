# Capture-recapture models in Stan

This repo provides efficient, flexible Stan implementations of capture-recapture (CR) models for ecologists and quantitative biologists. It contains log likelihood functions and Stan programs for both Cormack-Jolly-Seber (conditioned on first capture) and Jolly-Seber (additionally modeling the entry process) models. Both are implemented across single- and multi-state configurations (including multievent models), with and without robust design.

Example Stan programs of each configuration are found in the `stan` folder, with the log likelihood functions found in accompanying `.stanfunctions` files. All functions are overloaded to accommodate either (1) parameters varying by survey (or secondary) only or (2) parameters varying by individual as well. If no individual effects are required, the former is considerably faster, especially for Jolly-Seber models where the log likelihood of an augmented individual only has to be computed once. Jolly-Seber models feature additional "collapsed" function signatures, i.e. `js*2` and `js*2_rng()`, that accommodate individual effects for observed individuals but only one log likelihood computation for augmented individuals. All functions were written as efficiently as possible to allow the user to focus on flexibly modeling the model parameters without having to adjust the individual log likelihood computations.

All models feature the following by default:

1.  Mortality hazard rates and transition rates instead of probabilities to accommodate unequal survey intervals;

2.  In Jolly-Seber models, time-varying entry probabilities with an offset for survey length to accommodate unequal survey intervals, with options to use a Dirichlet or logistic-normal entry process;

3.  Individual log likelihoods stored in the `log_lik` variable to accommodate PSIS-LOO with the [loo](https://github.com/stan-dev/loo) package, and the prior log density stored in the `lprior` variable to accommodate prior sensitivity analysis with the [priorsense](https://github.com/n-kall/priorsense) package.

Additionally, all Jolly-Seber models feature `_rng` Stan functions in the `js-rng.stanfunctions` file that return population sizes ($\boldsymbol{N}$), number of entries ($\boldsymbol{B}$), and number of exits ($\boldsymbol{D}$) per survey, as well as the super-population ($N_\mathrm{super}$). In multistate Jolly-Seber models, these quantities are returned by each state.

All configurations were tested with simulation-based calibration (SBC), available in the `sbc` folder.

The `case-studies` folder contains code and Stan programs to run the examples in the manuscript entitled "An overview of capture–recapture models with efficient Bayesian implementations in Stan".
