#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kerkyra Asvesta
"""
"""
In this PYTHON file I show the MCMC analysis to evaluate the likelihood of the parameters in question for 
the standard ΛCDM model using the Pantheon SNIA data.

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
import emcee
from chainconsumer import ChainConsumer
from chainconsumer.diagnostic import Diagnostic
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


### ΛCDM q parameterization for the whole sample ###
def q_LCDM(z, Om):    ### q in LCDM model with w=-1 
    #up = Om*((1+z)**3)+(1+3*w)*(1-Om)*(1+z)**(3*(1+w))
    up = (Om*(1+z)**3) - (2*(1-Om))
    #down = Om*((1+z)**3)+(1-Om)*((1+z)**(3*(1+w)))
    down = 2*(Om*((1+z)**3) + 1-Om)
    return up/(down)


### Hubble parameter ###
def H_z(z,Om):
    f = lambda x: (1+q_LCDM(x, Om))/(1+x)
    integ = quad(f, 0.0, z)[0]
    ex = np.exp(integ)
    return ex*(H0)


### luminosity distance ###
def dl(z, Om):
    f = lambda y: (1/(H_z(y,Om)))
    inte = np.real(quad(f, 0.,z)[0])
    return inte*c_km*(1+z)

### Hubble-free luminosity distance ###
def Dl(z, Om):
    return (H0/c_km)*(dl(z, Om))


### the theoretical apparent magnitude ###
def m_z(z, Om, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl(z[i], Om)) + Mcal for i in range(size)])

### construct the chi squared function ###
def chi(params):
    Om = params[0]
    Mcal = params[1]
    res1 = np.array([m_z(z_cmb, Om, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_z(z_cmb, Om, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))


#########################
#                       #
#    MCMC               #
#                       #
######################### 


### define the prior

###   Set up a flat prior on a paramater.
###    Out of the bounds, the prior return -infinity, else it returns 0.
def lnprior_lcdm(theta):
    om, Mcal = theta
    if 0.1 < om < 0.9 and 23.0 < Mcal < 24.0:
        return 0.0
    return -np.inf

### combine into full log-probability function
def lnprob_lcdm(theta):
    lp = lnprior_lcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + (-0.5*chi(theta))

### define no of parameters and walkers
### ndiml is the number of dimensions, i.e. the number of free parameters
ndiml, nwalkersl = 2, 100
### initial values
initial = np.array([0.3, 23.8])
posl = [np.array(initial) + 0.0001*np.random.randn(ndiml) for i in range(nwalkersl)]
### run mcmc using multiprocessing
### Run MCMC with Multiple Cores
with Pool() as pool:
    samplerl = emcee.EnsembleSampler(nwalkersl, ndiml, lnprob_lcdm, pool=pool)
    resultl = samplerl.run_mcmc(posl, 2000, progress = True) #2000 is the number of MCMC steps to take
### The samplerl will give an array with thes shape (2000, 100, 2) giving the parameter values for each walker at each step in the chain.

### Get autocorrelation time
om_ac, M_cal_ac = samplerl.get_autocorr_time(quiet=True)
np.savetxt("fullflatchains_lcdm_2000", samplerl.get_chain(flat=True))
# thin out the chain
flat_samplesl = samplerl.get_chain(discard=20, flat=True, thin=int(max([om_ac, M_cal_ac])))
np.savetxt("flatchains_lcdm_2000 ", flat_samplesl)

print("Number of independent samples is {}".format(len(flat_samplesl)))

### Retrieve the flatchains from the MCMC analysis discarded the burn-in steps 
flatchains2000 = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q_LCDM/flatchains_lcdm_2000")

params = ["Om", "Mcal"]
### plot the chains for each parameter 
for i in range(2):
    plt.figure()
    plt.title(params[i])
    x = np.arange(len(flatchains2000[:,i]))

    plt.plot(x,flatchains2000[:,i])
    plt.show()

### USE CHAIN CONSUMER TO GENERATE PLOTS ###
cc = ChainConsumer()
params = ["Om", "Mcal"]
c = cc.add_chain(flatchains2000[:,:], parameters=params)
c.configure(contour_labels="confidence", kde= True, statistics="max",summary = False, shade_alpha=0.9,tick_font_size=11, label_font_size=15, sigmas=[1, 2], linewidths=1.2, colors="#673AB7", sigma2d = False, shade =True, flip = False)

###  The maximum likelihood values for the ΛCDM model can be found using the Analysis class of the Chain Consumer
BF_LCDM = c.analysis.get_summary().values()
bf_om = list(BF_LCDM)[0][1]
bf_mcal = list(BF_LCDM)[1][1]
truth = [bf_om, bf_mcal] 
fig = c.plotter.plot(figsize=2.0, extents=[[0.16, 0.40], [23.75, 23.875]], display=True, truth=truth)
