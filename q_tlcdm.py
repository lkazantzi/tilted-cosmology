#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kerkyra Asvesta

"""
### import python modules ###

import numpy as np
from numpy import *
from scipy.integrate import quad
import warnings
from math import sqrt
import matplotlib.pyplot as plt
from scipy import interpolate, linalg, optimize
from scipy.sparse.linalg import inv
from numpy.linalg import multi_dot
import scipy.optimize as opt
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize
import time
import lmfit
from lmfit import Parameters, fit_report, minimize, Minimizer
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from matplotlib import rcParams as rcp
from matplotlib import rc
plt.rcParams['figure.figsize'] = [9, 6]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 0
rcp['axes.formatter.useoffset'] = False
rcp['axes.linewidth'] = 1.5
rcp['axes.axisbelow'] = False
rcp['xtick.major.size'] = 8
rcp['xtick.minor.size'] = 4
rcp['xtick.labelsize'] = 15
rcp['legend.fontsize'] = 15
rcp['xtick.direction'] = 'in'
rcp['ytick.major.width'] = 2
rcp['ytick.minor.width'] = 2


c_km = (c.to('km/s').value)
H0 = 70.
cosmo.h
######## PANTHEON data   ######################
# import redshifts (cosmological + heliocentric), apparent magnitudes, error of app.magnitudes, systematic errors
######## PANTHEON data   ######################
# import redshifts (cosmological + heliocentric), apparent magnitudes, error of app.magnitudes,systematic errors

# Get the data from the github repository Pantheon of Dan Scolnic #
# https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long_zhel.txt #
data = np.loadtxt('/home/kerky/anaconda3/SN1A DATA/PANTHEON_DATA/Scolnic_data_updated.txt', usecols=[1,2,4,5])

### list with all the systematics as found in the PANTHEON catalog ###

# get the full systematics from the same repository #
#https://github.com/dscolnic/Pantheon/blob/master/sys_full_long.txt #
sys = np.genfromtxt('/home/kerky/anaconda3/SN1A DATA/PANTHEON_DATA/systematics.txt', skip_header=1)

### The list sn_names contains all the supernovae names in the PANTHEON catalog ###
sn_names = np.genfromtxt('/home/kerky/anaconda3/SN1A DATA/PANTHEON_DATA/Scolnic_data_updated.txt', usecols=[0],dtype='str')

z_cmb= np.array(data[:,0]) ## CMB redshift

z_hel = np.array(data[:,1]) ## heliocentric redshift

mb = np.array(data[:,2]) ## apparent magnitude

dmb = np.array(data[:,3]) ## error on mb

sqdmb = np.square(dmb) ## array with the square of the mb errors

### Construction of the full systematic covariance matrix ###
st = np.array(sys) ## put the systematics in an array
cov = np.reshape(st, (1048, 1048)) ## build the covariance matrix
er_stat = np.zeros((1048, 1048), float)
er_stat[np.diag_indices_from(er_stat)] = sqdmb
er_total = er_stat + cov
diag = np.array([np.diagonal(er_total)])
size = len(z_cmb)
print(size)
C_covn = np.linalg.inv([er_total]).reshape(1048, 1048)### the covariance matrix

### Construct q parametrizations for the three models : ( t-Λ ), ( t-Λ (α fixed) ) and ΛCDM ####

### Deceleration parameter in the tilted model using the ΛCDM comoving distance ( t-Λ ) ###
def integrand(z) : ## substituting Omega_m = 0.3 and Omega_l = 0.7
    return np.sqrt(1/(0.3*((1+z)**3) + 0.7))
def q_param(z, a, b):
    ratio = 1/(a+(b*(quad(lambda r:integrand(r), 0., z)[0]**3)))
    return 0.5*(1-ratio)
qv_lcdm = np.vectorize(q_param)

###  Deceleration parameter in the tilted model using the ΛCDM comoving distance & 
# fixed value of the parameter a (a = 0.5) ( t-Λ (α fixed) )
def q_aparam(z, b):
    ratio = 1/(0.5+(b*(quad(lambda r:integrand(r), 0., z)[0]**3)))
    return 0.5*(1-ratio)
qva_lcdm = np.vectorize(q_aparam)


### standard ΛCDM q parametrization ###
def q_LCDM(z, Om, w=-1):    ### q in LCDM model with w=-1 
    #up = Om*((1+z)**3)+(1+3*w)*(1-Om)*(1+z)**(3*(1+w))
    up = (Om*(1+z)**3) - (2*(1-Om))
    #down = 2*(Om*((1+z)**3)+(1-Om)*((1+z)**(3*(1+w))))
    down = 2*(Om*((1+z)**3) + 1-Om)
    return up/(down)
print(q_LCDM(0.0, 0.3))


### Hubble parameter in the ( t-Λ ) model ###
def H_param(z,a, b):
    f = lambda x: (1+q_param(x,a, b))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)

### Hubble parameter in the ( t-Λ (α fixed) ) model ###
def H_aparam(z, b):
    f = lambda x: (1+q_aparam(x, b))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)

### Hubble parameter in ΛCDM model ###
def H_LCDM(z, Om):
    f = lambda x: (1+q_LCDM(x, Om))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)


### luminosity distance in the ( t-Λ ) model ###
def dl_param(z,a,b):
    f = lambda y: (1/(H_param(y,a, b)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)



### luminosity distance in the ( t-Λ (α fixed) ) model ###
def dl_aparam(z, b):
    f = lambda y: (1/(H_aparam(y, b)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)

### luminosity distance in the ΛCDM model ###
def dl_LCDM(z, Om):
    f = lambda y: (1/(H_LCDM(y, Om)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)
### Hubble-free luminosity distance in the ( t-Λ ) model ###
def Dl_param(z,a, b):
    return (H0/c_km)*(dl_param(z,a, b))

### Hubble-free luminosity distance in the ( t-Λ (α fixed) ) model ###
def Dl_aparam(z, b):
    return (H0/c_km)*(dl_aparam(z, b))

### Hubble-free luminosity distance in the ΛCDM model ###
def Dl_LCDM(z, Om):
    return (H0/c_km)*(dl_LCDM(z, Om))

### Construct the theoretical apparent magnitude of the ( t-Λ ) model  ###
def m_theparam(z,a, b, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl_param(z[i],a, b)) + Mcal for i in range(size)])
print(m_theparam(z_cmb, 0.52, 3.7, 23.815).min())
print(m_theparam(z_cmb, 0.52, 3.7, 23.815).max())
print('The observed mb min is: ' ,mb.min())
print('The observed mb max is:' ,mb.max())

### Construct the theoretical apparent magnitude of the ( t-Λ (α fixed) ) model ###
def m_atheparam(z, b, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl_aparam(z[i], b)) + Mcal for i in range(size)])

### Construct the theoretical apparent magnitude of the ΛCDM model ###
def m_LCDM(z, Om, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl_LCDM(z[i], Om)) + Mcal for i in range(size)])


### construct the chi squared function for the ( t-Λ ) model ###
# remind that the chi squared is given by the relation (3.5) of the article
# the tilted ( t-Λ ) model contains 3 free parameters, a, b, Mcal 
def chi_param(params):
    a = params["a"].value
    b = params["b"]. value
    Mcal = params["Mcal"].value
    res1 = np.array([m_theparam(z_cmb, a, b, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_theparam(z_cmb, a, b, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))

### construct the chi squared function for the ( t-Λ (α fixed) ) ###
# the tilted ( t-Λ (α fixed) ) model contains 2 free parameters, b, Mcal
def chi_aparam(aparams):
    b = aparams["b"].value
    Mcal = aparams["Mcal"].value
    res1 = np.array([m_atheparam(z_cmb, b, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_atheparam(z_cmb, b, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))

### construct the chi squared function for the ΛCDM ###
# the ΛCDM model contains 2 free parameters, Om, Mcal
def chi_LCDM(lparams):
    Om = lparams[0]
    Mcal = lparams[1]
    res1 = np.array([m_LCDM(z_cmb, Om, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_LCDM(z_cmb, Om, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))


### FITTING PROCESS with lmfit for the ( t-Λ ) model ###
params = Parameters()
params.add("a", value = 0.5)
params.add("b", value = 3., min = 0., max = 30)
params.add("Mcal", value = 23.8, min = 23., max = 24.)
fitter = Minimizer(chi_param, params)
result = fitter.minimize(method="SLSQP", tol=10**-14)
result.residual # this is the value of the chi squared
# array([1026.69231994]) 
print(fit_report(result))
'''
[[Fit Statistics]]
    # fitting method   = SLSQP
    # function evals   = 109
    # data points      = 1
    # variables        = 3
    chi-square         = 1054097.12
    reduced chi-square = 1054097.12
    Akaike info crit   = 19.8681951
    Bayesian info crit = 13.8681951
##  Warning: uncertainties could not be estimated:
[[Variables]]
    a:     0.52660166 (init = 0.5)
    b:     3.71009900 (init = 3)
    Mcal:  23.8155471 (init = 23.8)
'''
BF = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q(l)_LCDM_1048/deMcal_1048_LCDM ")
#BF[0] = 0.52660166
#BF[1] = 3.71009900
#BF[2] = 23.8155471

### FITTING PROCESS for the ( t-Λ (α fixed) ) ###
aparams = Parameters()
aparams.add("b", value = 5., min = 0., max = 30)
aparams.add("Mcal", value = 23.80, min = 23., max = 24.)
fittera = Minimizer(chi_aparam, aparams)
result_a = fittera.minimize(method="SLSQP", tol=10**-14)
result_a.residual  # this is the value of the chi squared
# array([1027.20693316])
print(fit_report(result_a)) 
'''
[[Fit Statistics]]
    # fitting method   = SLSQP
    # function evals   = 42
    # data points      = 1
    # variables        = 2
    chi-square         = 1055154.08
    reduced chi-square = 1055154.08
    Akaike info crit   = 17.8691974
    Bayesian info crit = 13.8691974
##  Warning: uncertainties could not be estimated:
[[Variables]]
    b:     5.20420255 (init = 5)
    Mcal:  23.8072574 (init = 23.8)
'''
BFa = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q(l)_LCDM_1048/bMcal_1048_LCDMa")
#BFa[0] = 5.20420255
#BFa[1] = 23.8072574

### FITTING PROCESS for the ΛCDM model ###
lguess = np.array([0.3, 23.8])
res = opt.minimize(chi_LCDM, lguess, method= 'SLSQP',  tol=10**-14)
print(res)
'''
fun: 1026.670550690294
    jac: array([ 2.13623047e-04, -3.05175781e-05])
message: 'Optimization terminated successfully'
   nfev: 44
    nit: 8
   njev: 8
 status: 0
success: True
      x: array([ 0.29948314, 23.80931084])
''' 
BFlcdm = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q_LCDM/OmMcal_1048_LCDM ")
#BFlcdm[0] = 0.29948322


### Plot of the t-Λ, t-Λ (α fixed) and standard LCDM models ###
fig, ax = plt.subplots()
l = np.linspace(0.0, 6, 900)
plt.plot(l,qv_lcdm(l,BF[0], BF[1]), color = 'yellow', label=(r"$\frac{ 1 } { 2 }\left(1-\frac{1}{0.5+b(\chi_{\Lambda CDM})^3}\right)$"))

plt.plot(l,qva_lcdm(l,BFa[0]), color = 'r',label=(r"$\frac{ 1 } { 2 }\left(1-\frac{1}{a+b(\chi_{\Lambda CDM})^3}\right)$"))

plt.plot(l,q_LCDM(l, BFlcdm[0]), color = 'b', label='qLCDM')

plt.rc('text', usetex=True)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.xlabel('redshift (z)', loc='right')
ax.set_xticks([0.6, 1,2,3,4, 5, 6])
ax.set_yticks([-0.5,0, 0.2, 0.5])
ax.legend(fontsize = 'large', loc="lower right")
plt.show()

###########################
### RESULTS OF TABLE II ###
###########################

### AIC + BIC criteria for model selection between the models ( ΛCDM + t-Λ ) , ( ΛCDM + t-Λ (α fixed) ) ###
### For the ΛCDM ###
AIC_qlcdm = BFlcdm[2] + 2*2
BIC_qlcdm = BFlcdm[2] + np.log(len(mb)) * 2
print(AIC_qlcdm, BIC_qlcdm)
#AIC =  1030.6705564254162 and BIC = 1040.579834155178

### For the t-Λ model ###
AIC_tl = BF[3] + 2*3
BIC_tl = BF[3] + np.log(len(mb)) *3
print(AIC_tl, BIC_tl)
#AIC =  1032.6923199395605 and BIC = 1047.5562365342034

### For the t-Λ (α fixed) model ###
AIC_tlafxd = BFa[2] + 2*2
BIC_tlafxd = BFa[2] + np.log(len(mb)) *2
print(AIC_tlafxd, BIC_tlafxd)
#AIC =  1031.20693316 and BIC = 1041.116210889762

### AIC Differences (ΔAIC) ###
daictl = np.abs(AIC_tl - AIC_qlcdm)
daictlafxd = (AIC_tlafxd - AIC_qlcdm)
print(daictl, daictlafxd)
# "The AIC difference between the t-Λ and the ΛCDM is:", 2.021763514144368
# "The AIC difference between the t-Λ (α fixed) and the ΛCDM is:", 0.5363767345838824

### BIC Differences (ΔBIC) ###
dbictl = np.abs(BIC_tl - BIC_qlcdm)
dbictlafxd = (BIC_tlafxd - BIC_qlcdm)
print(dbictl, dbictlafxd)
# "The BIC difference between the t-Λ and the ΛCDM is:",  6.9764023790253304
# "The BIC difference between the t-Λ (α fixed) and the ΛCDM is:", 0.5363767345838824

 
