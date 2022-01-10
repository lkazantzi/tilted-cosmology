# Observational Constraints on the Deceleration Parameter in the Tilted Frame

[![Travis](https://img.shields.io/badge/language-Mathematica-green.svg)]()
[![Travis](https://img.shields.io/badge/language-Python-yellow.svg)]()


<p align="center">
<img src="preview.png" width="900" title="preview" />
</p>

This is the repository that contains part of the codes as well as useful comments that reproduce the figures of our project. The zip folder CLASS Implementations only contains the files that are modified in order to implement the Transition Models in the CLASS and MontePython programs. In particular, the default `background.c`, `input.c` and `background.h` should be substituted with the corresponding files of the zip folder. Furthermore, the `Pantheon_SN` folder should be added to the default `.../montepython/likelihoods` folder in order to correctly implement the transition on the absolute magnitude *M* at z<sub>t</sub>. The files `Fig_1.nb` and `Absolute_Magnitude_BestFit.nb` correspond to the Mathematica codes for the construction of Fig. 1 and Fig. 7 respectively. Finally, in order to estimate the statistical significance of our results we used the Akaike Information Criterion (AIC) as well as the [MCEvidence](https://github.com/yabebalFantaye/MCEvidence) package. For the AIC, we derived the corresponding differences using the Mathematica file `AIC_Calculations.nb`.

## Abstract
...


## Instructions - CLASS Implementation
....


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
