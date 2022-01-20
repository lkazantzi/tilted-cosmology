# Observational Constraints on the Deceleration Parameter in the Tilted Frame

[![Travis](https://img.shields.io/badge/language-Mathematica-green.svg)]()
[![Travis](https://img.shields.io/badge/language-Python-yellow.svg)]()


<p align="center">
<img src="preview.png" width="820" title="preview" />
</p>

This is the repository that contains the corresponding Mathematica and Python files as well as useful comments that reproduce the figures of our work. 


## Abstract
...


## Instructions
### Mathematica Files
Regarding the relevant Mathematica codes, the file `Tilted Cosmology.nb` corresponds to the Mathematica code in order to obtain the results of the maximum likelihood method for all the models described in Table I. The file `Information Criteria.nb` corresponds to the calculations of Table II. The best fit results were obtained with Mathematica 11. 

### Python Files
Regarding the relevant Python codes, the files `q_tEdS.py` and `q_tlcdm.py` contain the functions and the optimisation methods developped for calculating the best-fit parameters for the tilted Einstein-de Sitter and the tilted ΛCDM model respectively. The files above contain also the calculations for the information criteria, in Table II. 
few words about the MCMC analysis..
We follow a Bayesian approach and employ the Markov Chain Monte Carlo (MCMC) method to estimate the posterior probability of the model parameterd . We use the emcee python-based MCMC package  (https://github.com/dfm/emcee). For all of our cosmological models, we generate 100 walkers with 2000 steps and use multiprocessing to run the walkers in parallel. The MCMC analysis for the tilted EdS bulk flow model (t-EdS) is found in the qt_EdS_MCMC.py file. The analysis for the tilted EdS model when the parameter a is fixed (t-EdS (a fixed)) is found in the qt_EdSa_MCMC.py file. The analysis for the standard ΛCDM model is in the q_LCDM_MCM.py file. We use flat priors defined in the prior function and instantiate each walker with a starting point close to the maximum-likelihood point from the minimization process. The Python libraries needed to run the files are reported in the `requirements.txt`. 


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
