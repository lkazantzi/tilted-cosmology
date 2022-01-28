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
Regarding the relevant Python codes, the files `q_tEdS.py` and `q_tlcdm.py` contain the functions and the optimisation methods developped for calculating the best-fit parameters for the tilted Einstein-de Sitter and the tilted ΛCDM model respectively. The files above contain also the calculations for the information criteria, in Table II. The codes for the reproduction of the Figure 1 of the article are found in the files `q_tEdS.py` (for the left panel) and `q_tlcdm.py` (for the right panel). 

*Few words about the MCMC analysis*

We follow a Bayesian approach and employ the Markov Chain Monte Carlo (MCMC) method to estimate the posterior probability of the models parameters and their uncertainties. We use the [emcee](https://github.com/dfm/emcee) python-based MCMC package. For all of our cosmological models, we use 100 random walkers to explore the entire parameter space with 2000 iteretations (steps) and use the method of multiprocessing to run the walkers in parallel. We use flat priors defined in the prior function and instantiate each walker with a starting point close to the maximum-likelihood point from the minimization process. When the run is completed, using the [ChainConsumer](https://samreay.github.io/ChainConsumer/) package, we do the triangle plot to visualise the posterior distribution of the parameters of each model together with their 1-2σ confidence levels. We use Matplotlib library to generate the trace plots of every parameter of our models. The file 'q_LCDM_MCMC.py' contains the code for performing an MCMC analysis for the concordance ΛCDM model. Also contains code to compute confidence levels of the posterior distributions and generate the contour plot. The files 'qt_EdS_MCMC.py' and 'qt_EdSa_MCMC.py' contain the codes for the MCMC analysis of the tilted Einstein-de Sitter model and the Einstein-de Sitter model with the parameter α fixed, respectively. The code for the Figure 2 is in the 'qt_EdS_MCMC.py' file. The files 'qt_LCDM_MCMC.py' and 'qt_LCDMa_MCMC.py' contain the MCMC analysis for the tilted ΛCDM and the tilted ΛCDM with α fixed models respectively. The code for the Figure 3 is in the 'qt_LCDM_MCMC.py' file. 
The Python libraries needed to run the files are reported in the `requirements.txt`. 


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
