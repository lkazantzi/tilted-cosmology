#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kerkyra Asvesta
In this PYTHON file I show the MCMC analysis to evaluate the likelihood of the parameters in question for 
the  tilted Einstein-de Sitter model with the parameter a is fixed (t-EdS (a fixed))
"""

import multiprocessing as mproc
from multiprocessing import Pool
import numpy as np
from numpy import *
from scipy.integrate import quad
from math import sqrt
import matplotlib.pyplot as plt
from scipy import interpolate, linalg, optimize
from scipy.sparse.linalg import inv
from numpy.linalg import multi_dot
from astropy.constants import c
from chainconsumer import ChainConsumer
from matplotlib import rcParams as rcp

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
rcp['font.size'] = 20

### some constants ###
c_km = (c.to('km/s').value)
H0 = 70.

####################################
#                                  #
#    import the Pantheon SNIa data #
#                                  #
####################################

# import redshifts (cosmological + heliocentric), apparent magnitudes, error of app.magnitudes, systematic errors
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

### construct a q parameterization with respect to the Einstein de Sitter comoving distance ###
def dc_eds(z): ### comoving distance in EdS Universe
    return 2*(1-((1+z)**(-1/2)))

### q parameterization wrt the EdS comoving distance when a is fixed###
def q_edsa(z,b):
    return 0.5 *(1 -(1/(((np.power(dc_eds(z), 3))*b)+0.5)))

### Hubble parameter in EdS  ###
def H_edsa(z,b):
    f = lambda x: (1+q_edsa(x, b))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)

### luminosity distance in EdS ###
def dl_edsa(z,b):
    f = lambda y: (1/(H_edsa(y, b)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)


### Hubble-free luminosity distance in EdS ###
def Dl_edsa(z, b):
    return (H0/c_km)*(dl_edsa(z,b))

### the theoretical apparent magnitude in EdS ###
def m_edsa(z,b, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl_edsa(z[i], b)) + Mcal for i in range(size)])


### construct the chi squared function in EdS ###
def chi_edsa(params):
    b = params[0]
    Mcal = params[1]
    res1 = np.array([m_edsa(z_cmb, b, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_edsa(z_cmb, b, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))


#########################
#                       #
#    MCMC               #
#                       #
######################### 

### Define the prior function
def lnprior_eds(theta):
    b, Mcal = theta
### define priors 
    if 0.0 < b < 35 and 23.0 < Mcal < 24.0:
        return 0.0
    return -np.inf
print(lnprior_eds(theta))

### combine into full log-probability function
def lnprob_eds(theta):
    """
    Return the log posterior for given theta
    """ 
    lp = lnprior_eds(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + (-0.5*chi_edsa(theta))

### test the result of the posterior
print(lnprob_eds(theta))
### define no of parameters (equals the no of dimensions) and walkers
ndiml, nwalkersl = 2, 100
### define initial positions
### Choose the start location relatively close to the maximum-likelihood point from the minimization process
initial = np.array([8.47 23.81])
posl = [np.array(initial) + 0.0001*np.random.randn(ndiml) for i in range(nwalkersl)]
### run mcmc using multiprocessing to reduce the calcullation time 
with Pool() as pool:
    #Define the Affine-Invariant chain
    sampler_eds = emcee.EnsembleSampler(nwalkersl, ndiml, lnprob_eds, pool=pool)
    result_eds = sampler_eds.run_mcmc(posl, 2000, progress = True) # 2000 is the no of steps
#### Calculate the autocorrelation length (ACL) (or time). 
####You can use this number to thin out chains to produce independent samples.
b_ac, M_cal_ac = sampler_eds.get_autocorr_time(quiet=True)
print("The autocorrelation length for a is {0} and b is {1} and M_cal is {2}".format(b_ac, M_cal_ac))

np.savetxt("fullflatchains_edsa_2000 ",sampler_eds.get_chain(flat=True))
### thin out the chain
flat_samples_eds = sampler_eds.get_chain(discard=20, flat=True, thin=(int(max[b_ac, M_cal_ac])))
np.savetxt("flatchains_edsa_2000 ",flat_samples_eds)

print("Number of independent samples is {}".format(len(flat_samples_eds)))

### The flatchains from the MCMC analysis discarded the burn-in steps 
flatchains_edsa = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q(l)_EdS_1048/flatchains_edsa_2000")

### plot the chains for each parameter 
params = ["b", "Mcal"]
for i in range(2):
    plt.figure()
    plt.title(params[i])
    x = np.arange(len(flatchains_edsa[:,i]))
    plt.plot(x,flatchains_edsa[:,i])
    plt.show()



### USE CHAIN CONSUMER TO GENERATE PLOTS ###

params = [r"b", r"${\cal{M}}$"]
cc = ChainConsumer()
c = cc.add_chain(flatchains_edsa[:,:], parameters=params)
c.configure(statistics = "max", kde= True, max_ticks=7, summary = False, shade_alpha=0.9, tick_font_size=11, label_font_size=15, sigmas=[1, 2], linewidths=1.2, colors="#673AB7", sigma2d = False, shade =True, flip = False)
###  The maximum likelihood values for the t - EdSa model can be found using the Analysis class of the Chain Consumer
BF_EdSa = c.analysis.get_summary().values()
bf_b = list(BF_EdSa)[0][1]
bf_mcal = list(BF_EdSa)[1][1]
print(bf_b, bf_mcal)

truth = [bf_b, bf_mcal]
fig = c.plotter.plot(figsize=2.0, extents=[[0, 25], [23.75, 23.875]], display=True, truth = truth)

