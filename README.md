# Observational Constraints on the Deceleration Parameter in the Tilted Frame

[![Travis](https://img.shields.io/badge/language-Mathematica-green.svg)]()
[![Travis](https://img.shields.io/badge/language-Python-yellow.svg)]()


<p align="center">
<img src="preview.png" width="820" title="preview" />
</p>

This is the repository that contains the corresponding Mathematica and Python files as well as useful comments that reproduce the figures of our work. 


## Abstract
We study a parametrization of the deceleration parameter in a tilted universe, namely a cosmological model equipped with two families of observers. The first family follows the smooth Hubble flow, while the second are the real observers residing in a typical galaxy inside a bulk flow and moving relative to the smooth Hubble expansion with finite peculiar velocity. We use the compilation of Type Ia Supernovae (SnIa) data, as described in the Pantheon dataset, to find the quality of fit to the data and study the redshift evolution of the deceleration parameter. In so doing, we consider two alternative scenarios, assuming that the bulk-flow observers live in the ΛCDM and in the Einstein-de Sitter universe. We show that a tilted Einstein-de Sitter model can reproduce the recent acceleration history of the universe, without the need of a cosmological constant or dark energy, but by simply accounting for the linear effects of peculiar motions. By means of a Markov Chain Monte Carlo (MCMC) method, we also constrain the magnitude and the uncertainties of the parameters of the two models. From our statistical analysis, we find that the tilted Einstein-de Sitter model, equipped with one or two additional parameters that describe the assumed large-scale velocity flows, performs similar to the standard ΛCDM paradigm in the context of model selection criteria (Akaike Information Criterion and Bayesian Information Criterion). 


## Instructions
### Mathematica Files
Regarding the relevant Mathematica codes, the file `Tilted Cosmology.nb` corresponds to the Mathematica code in order to obtain the results of the maximum likelihood method for all the models described in Table I. The file `Information Criteria.nb` corresponds to the calculations of Table II. The best fit results were obtained with Mathematica 11. 

### Python Files
Regarding the relevant Python codes, the files `q_tEdS.py` and `q_tlcdm.py` contain the functions and the optimisation methods developped for calculating the best-fit parameters for the tilted Einstein-de Sitter and the tilted ΛCDM model respectively. The files above contain also the calculations for the information criteria, in Table II. 

*Few words about the MCMC analysis*

We follow a Bayesian approach and employ the Markov Chain Monte Carlo (MCMC) method to estimate the posterior probability of the models parameters. We use the [emcee](https://github.com/dfm/emcee) python-based MCMC package. For all of our cosmological models, we generate 100 walkers with 2000 steps and use multiprocessing to run the walkers in parallel. We use flat priors defined in the prior function and instantiate each walker with a starting point close to the maximum-likelihood point from the minimization process. When the run is completed, using the [ChainConsumer](https://samreay.github.io/ChainConsumer/) package, we do the triangle plot to study the posterior distribution of the parameters of each model and the 1-2σ confidence levels. We use Matplotlib library to generate the trace plots of every parameter of our models. The Python libraries needed to run the files are reported in the `requirements.txt`. 


## Citing the paper 
If you use any of the above codes or the figures in a published work please cite the following paper:
<br>*Observational Constraints on the Deceleration Parameter in the Tilted Frame*
<br>Kerkyra Asvesta, Lavrentios Kazantzidis, Leandros Perivolaropoulos, and Christos Tsagas

Any further questions/comments are welcome.


## Authors List
<br>Kerkyra Asvesta - <keasvest@auth.gr>
<br>Lavrentios Kazantzidis - <l.kazantzidis@uoi.gr>
<br>Leandros Perivolaropoulos - <leandros@uoi.gr>
<br>Christos Tsagas - <tsagas@astro.auth.gr>
