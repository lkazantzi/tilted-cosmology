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
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) #from astropy.cosmology import FlatLambdaCDM
from matplotlib import rcParams as rcp
from matplotlib import rc
rcp['figure.figsize'] = [9, 6]
rcp['figure.dpi'] = 80
rcp['xtick.labelsize'] = 15
rcp['ytick.labelsize'] = 15
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
print(c_km)
H0 = 70.
q0 = -0.55
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
cosmo.Om0
print(cosmo.h)

###############################################
#                                             #
###            SNeIa PANTHEON data          ###
#                                             #
###############################################
# import redshifts (cosmological + heliocentric), apparent magnitudes, error of app.magnitudes, systematic errors
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

### standard ΛCDM q parametrization ###
def q_LCDM(z, Om, w=-1): # with w the equation of state
    #up = Om*((1+z)**3)+(1+3*w)*(1-Om)*(1+z)**(3*(1+w))
    up = (Om*(1+z)**3) - (2*(1-Om))
    #down = Om*((1+z)**3)+(1-Om)*((1+z)**(3*(1+w)))
    down = 2*(Om*((1+z)**3) + 1-Om)
    return up/(down)
print(q_LCDM(0.0, 0.3))

### construct the comoving distance of the Einstein-de Sitter (EdS) model ###
# Multiply with H0/c_km in order for the deceleration parameter to be dimensionless
def dc_eds(z):
    #return (2*c_km/H0)*(1-((1+z)**(-1/2)))
    return 2*(1-((1+z)**(-1/2)))
    #return 2*((1+z) - ((1+z)**(1/2)))


### Deceleration parameter in the tilted model using the EdS comoving distance ( t-EdS ) ###
def q_eds(z,a,b):
    return 0.5 *(1 -(1/(((np.power(dc_eds(z), 3))*b)+a)))
qv_eds= np.vectorize(q_eds)


### Deceleration parameter in the tilted model using the EdS comoving distance & 
# fixed value of the parameter a (a = 0.5) ( t-EdS (α fixed) ) ###
def qa_eds(z,b):
    return 0.5 *(1 -(1/(((np.power(dc_eds(z), 3))*b)+0.5)))
qva_eds = np.vectorize(qa_eds)


### Hubble parameter in the ( t-EdS ) model ###
def H_eds(z,a, b):
    f = lambda x: (1+q_eds(x, a, b))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)

### Hubble parameter in the ( t-EdS (α fixed) ) ###
def Ha_eds(z, b):
    f = lambda x: (1+qa_eds(x, b))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)

### Hubble parameter in ΛCDM model ###
def H_LCDM(z, Om):
    f = lambda x: (1+q_LCDM(x, Om))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)

### luminosity distance in the ( t-EdS ) model ###
def dl_eds(z,a,b):
    f = lambda y: (1/(H_eds(y,a, b)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)

### luminosity distance in the ( t-EdS (α fixed) ) model ###
def dla_eds(z,b):
    f = lambda y: (1/(Ha_eds(y, b)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)


### luminosity distance in the ΛCDM model ###
def dl_LCDM(z, Om):
    f = lambda y: (1/(H_LCDM(y, Om)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)

### Hubble-free luminosity distance in the ( t-EdS ) model ###
def Dl_eds(z,a, b):
    return (H0/c_km)*(dl_eds(z,a, b))

### Hubble-free luminosity distance in the ( t-EdS (α fixed) ) model ###
def Dla_eds(z, b):
    return (H0/c_km)*(dla_eds(z,b))

### Hubble-free luminosity distance in the ΛCDM model ###
def Dl_LCDM(z, Om):
    return (H0/c_km)*(dl_LCDM(z, Om))

### the theoretical apparent magnitude of the ( t-EdS ) model ###
def m_eds(z,a, b, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl_eds(z[i],a, b)) + Mcal for i in range(size)])

### the theoretical apparent magnitude of the ( t-EdS (α fixed) ) model ###
def ma_eds(z, b, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dla_eds(z[i], b)) + Mcal for i in range(size)])

### Construct the theoretical apparent magnitude of the ΛCDM model ###
def m_LCDM(z, Om, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl_LCDM(z[i], Om)) + Mcal for i in range(size)])

### construct the chi squared function in the ( t-EdS ) model ###
# the tilted ( t-EdS ) model contains 3 free parameters, a, b, Mcal 
def chi_eds(params):
    a = params["a"].value
    b = params["b"].value
    Mcal = params["Mcal"].value
    res1 = np.array([m_eds(z_cmb, a, b, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_eds(z_cmb, a, b, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))

### construct the chi squared function in the ( t-EdS (α fixed) ) model ###
# the tilted ( t-EdS (α fixed)) model contains 2 free parameters, b, Mcal 
def chia_eds(aparams):
    b = aparams["b"].value
    Mcal = aparams["Mcal"].value
    res1 = np.array([ma_eds(z_cmb, b, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([ma_eds(z_cmb, b, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))

### construct the chi squared function for the ΛCDM ###
# the ΛCDM model contains 2 free parameters, Om, Mcal
def chi_LCDM(lparams):
    Om = lparams[0]
    Mcal = lparams[1]
    res1 = np.array([m_LCDM(z_cmb, Om, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_LCDM(z_cmb, Om, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))

### FITTING PROCESS with lmfit for the ( t-EdS ) model ###
params = Parameters()
params.add("a", value=0.5, min = 0.3, max = 0.8)
params.add("b", value = 6, min = 0.1, max = 30)
params.add("Mcal", value = 23.80, min = 23., max = 24.)
fitter = Minimizer(chi_eds, params)
result_eds = fitter.minimize(method="SLSQP")
print(result_eds.residual) # this is the value of the chi squared
#[1026.75607869]
print(fit_report(result_eds))
'''
[[Fit Statistics]]
    # fitting method   = SLSQP
    # function evals   = 39
    # data points      = 1
    # variables        = 3
    chi-square         = 1054228.05
    reduced chi-square = 1054228.05
    Akaike info crit   = 19.8683193
    Bayesian info crit = 13.8683193
##  Warning: uncertainties could not be estimated:
[[Variables]]
    a:     0.52064421 (init = 0.5)
    b:     6.65570490 (init = 6)
    Mcal:  23.8145347 (init = 23.8)
'''
### The maximum likelihood values for the t-EdS model are 
BF_eds = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q(l)_EdS_1048/abMcal_1048_EdS ")
#BF_eds[0] = 0.52064421
#BF_eds[1] = 6.65570490
#BF_eds[2] = 23.8145347

### FITTING PROCESS with lmfit for the ( t-EdS (α fixed)) model ###
aparams = Parameters()
aparams.add("b", value = 6, min = 0.1, max = 30)
aparams.add("Mcal", value = 23.80, min = 23., max = 24.)
fittera = Minimizer(chia_eds, aparams)
result_edsa = fittera.minimize(method="SLSQP")
print(result_edsa.residual) # this is the value for the chi squared for this model
#[1027.04505089]
print(fit_report(result_edsa))
'''
[[Fit Statistics]]
    # fitting method   = SLSQP
    # function evals   = 26
    # data points      = 1
    # variables        = 2
    chi-square         = 1054821.54
    reduced chi-square = 1054821.54
    Akaike info crit   = 17.8688822
    Bayesian info crit = 13.8688822
##  Warning: uncertainties could not be estimated:
[[Variables]]
    b:     8.47405255 (init = 6)
    Mcal:  23.8082143 (init = 23.8)
'''
### The maximum likelihood values for the t-EdS (α fixed) model are 
BF_edsa = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q(l)_EdS_1048/bMcal_EdSa ")
#BF_edsa[0] = 8.47410191 
#BF_edsa[1] =  23.8082156


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

### Plot of the t-EdS, t-EdS (α fixed) and standard LCDM models ###
fig, ax = plt.subplots()
l = np.linspace(0.0, 3, 900)
plt.plot(l,qv_eds(l,BF_eds[0], BF_eds[1]), color = 'r', label=(r"$\frac{ 1 } { 2 }\left(1-\frac{1}{a+b(\bar{\chi}_{EdS})^3}\right)$"))

plt.plot(l,qva_eds(l,BF_edsa[0]), color = 'yellow', label=(r"$\frac{ 1 } { 2 }\left(1-\frac{1}{0.5+b(\bar{\chi}_{EdS})^3}\right)$"))

plt.plot(l,q_LCDM(l, BFlcdm[0]), color = 'blue', linestyle = "--", label = (r"$q_{\Lambda CDM}$"))

plt.rc('text', usetex=True)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlabel('redshift (z)', loc='right')
ax.set_ylabel(' $\widetilde{q}$', loc='top', rotation = 0)
ax.set_yticks([-0.5, -0.4, -0.2, 0.2, 0.5])
ax.set_xticks([0.5, 1,1.5, 2])
ax.legend(fontsize = 'small', loc="lower right")
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

### For the t-EdS model ###
AIC_teds = BF_eds[3] + 2*3
BIC_teds = BF_eds[3] + np.log(len(mb)) *3
print(AIC_teds, BIC_teds)
#AIC =  1032.7560785796672 and BIC = 1047.6199951743101

### For the t-EdS (α fixed) model ###
AIC_tedsafxd = BF_edsa[2] + 2*2
BIC_tedsafxd = BF_edsa[2] + np.log(len(mb)) *2
print(AIC_tedsafxd, BIC_tedsafxd)
#AIC =  1031.0450508888207 and BIC = 1040.9543286185826

### AIC Differences (ΔAIC) ###
daicteds = np.abs(AIC_teds - AIC_qlcdm)
daictedsafxd = (AIC_tedsafxd - AIC_qlcdm)
print(daicteds, daictedsafxd)
# "The AIC difference between the t-Λ and the ΛCDM is:", 2.0855221542510662
# "The AIC difference between the t-Λ (α fixed) and the ΛCDM is:", 0.37449446340451686

### BIC Differences (ΔBIC) ###
dbicteds = np.abs(BIC_teds - BIC_qlcdm)
dbictedsafxd = (BIC_tedsafxd - BIC_qlcdm)
print(dbicteds, dbictedsafxd)
# "The BIC difference between the t-Λ and the ΛCDM is:",  7.040161019132029
# "The BIC difference between the t-Λ (α fixed) and the ΛCDM is:", 0.37449446340451686






