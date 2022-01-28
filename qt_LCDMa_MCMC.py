#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kerkyra Asvesta
In this PYTHON file I show the MCMC analysis to evaluate the likelihood of the parameters in question for 
the  tilted ΛCDM model with the parameter a is fixed (t-Λ (a fixed))
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

def integrand(z) : ## substituting Omega_m = 0.3 and Omega_l = 0.7
    return np.sqrt(1/(0.3*((1+z)**3) + 0.7))

###  Deceleration parameter in the tilted model using the ΛCDM comoving distance & 
# fixed value of the parameter a (a = 0.5) ( t-Λ (α fixed) )
def q_aparam(z, b):
    ratio = 1/(0.5+(b*(quad(lambda r:integrand(r), 0., z)[0]**3)))
    return 0.5*(1-ratio)

### Hubble parameter in the ( t-Λ (α fixed) ) model ###
def H_aparam(z, b):
    f = lambda x: (1+q_aparam(x, b))/(1+x)
    integ = quad(f, 0.0, z, epsabs=np.inf)[0]
    ex = np.exp(integ)
    return ex*(H0)

### luminosity distance in the ( t-Λ (α fixed) ) model ###
def dl_aparam(z, b):
    f = lambda y: (1/(H_aparam(y, b)))
    inte = np.real(quad(f, 0.,z, epsabs=np.inf)[0])
    return inte*c_km*(1+z)


### Hubble-free luminosity distance in the ( t-Λ (α fixed) ) model ###
def Dl_aparam(z, b):
    return (H0/c_km)*(dl_aparam(z, b))


### Construct the theoretical apparent magnitude of the ( t-Λ (α fixed) ) model ###
def m_atheparam(z, b, Mcal):
        #Mcal = M +42.38 - 5*np.log10(cosmo.h)
    return np.hstack([5*np.log10(Dl_aparam(z[i], b)) + Mcal for i in range(size)])


### construct the chi squared function for the ( t-Λ (α fixed) ) ###
# the tilted ( t-Λ (α fixed) ) model contains 2 free parameters, b, Mcal
def chi_aparam(aparams):
    b = aparams["b"].value
    Mcal = aparams["Mcal"].value
    res1 = np.array([m_atheparam(z_cmb, b, Mcal) - mb]).reshape(1, 1048)
    res3 = np.array([m_atheparam(z_cmb, b, Mcal) - mb]).reshape(1048, )
    return np.sum(np.linalg.multi_dot([res1, C_covn, res3]))

#########################
#                       #
#    MCMC               #
#                       #
######################### 

### define the prior ###
def lnpriorl(theta):
    b, Mcal = theta
    if  0.0 < b < 35 and 23. < Mcal < 24.0:
        return 0.0
    return -np.inf

### combine into full log-probability function
def lnprobl(theta):
    lp = lnpriorl(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + (-0.5*chia_param(theta))

### define no of parameters and walkers
### ndim defines the number of dimensions (parameters)
ndiml, nwalkersl = 2, 100

# create an initialization for the walkers
initial = np.array([ 5.8, 23.810 ])
posl = [np.array(initial) + 0.0001*np.random.randn(ndiml) for i in range(nwalkersl)]
### run mcmc using multiprocessing
with Pool() as pool:
    samplerl = emcee.EnsembleSampler(nwalkersl, ndiml, lnprobl, pool=pool)
    resultl = samplerl.run_mcmc(posl, 2000, progress = True) # 2000 is the no of steps
### generate the flat chains and discard the burn-in steps

full_flat_samplesl = samplerl.get_chain(flat=True)
np.savetxt("fullchains_qla_LCDM_2000 ",full_flat_samplesl)
b_ac, M_cal_ac = samplerl.get_autocorr_time(quiet=True)
print("The autocorrelation length for b is {0} and M_cal is {1}".format(b_ac, M_cal_ac))
flat_samplesl = samplerl.get_chain(discard=20, flat=True, thin=int(max([b_ac, M_cal_ac])))
np.savetxt("flatchains_qla_LCDM",flat_samplesl)

### The flatchains from the MCMC analysis discarded the burn-in steps 
flatchains_qla_LCDM = np.loadtxt("/home/kerky/anaconda3/test_q/resultsfromq(L)/q(l)_LCDM_1048/flatchains_qla_LCDM")
### the best-fit values of the tilted ΛCDM model (a fixed)(t-ΛCDM (a fixed))
BF_aLCDM = [5.204, 23.807]
params = ["b", "Mcal"]
### plot the chains for each parameter 
for i in range(2):
    plt.figure()
    plt.title(params[i])
    x = np.arange(len(flatchains_qla_LCDM[:,i]))
    plt.plot(x,flatchains_qla_LCDM[:,i])
    plt.axhline(y = BF_aLCDM[i], color = "red")
    plt.show()


 ### Compute the sigma errors ###

### Compute the sigma errors ###
def quantile(sorted_array, f):
    """Return the quantile element from sorted_array, where f is [0,1]
    using linear interpolation.

    Based on the description of the GSL routine
    gsl_stats_quantile_from_sorted_data - e.g.
    http://www.gnu.org/software/gsl/manual/html_node/Median-and-Percentiles.html
    but all errors are my own.

    sorted_array is assumed to be 1D and sorted.
    """
    sorted_array = np.asarray(sorted_array)

    if len(sorted_array.shape) != 1:
        raise RuntimeError("Error: input array is not 1D")
    n = sorted_array.size

    q = (n - 1) * f
    i = np.int64(np.floor(q))
    delta = q - i

    return (1.0 - delta) * sorted_array[i] + delta * sorted_array[i + 1]


### Calculate the 68%, 95%, 99.7% C.L. of the parameters
def get_error_estimates123(x, sorted=False, sigma="a"):
    """Compute the median and (-1,+1) sigma values for the data.
    Parameters
    ----------
    sigma defines the confidence interval. If default, the functions gives the 1sigma unvertainty
    for the input parameter. If sigma=2, then it gives the 2sigma uncertainty (~95%) and
    when sigma=3, then the function returns the 3sigma uncertainty(~99%) 
    
    Returns
    -------
    (median, lsig, usig)
       The median, value that corresponds to -1 sigma, and value that
       is +1 sigma, for the input distribution.

    Examples
    --------
    >>> (m, l, h) = get_error_estimates123(x, sigma="a")
    """
    xs = np.asarray(x)
    if not sorted:
        xs.sort()
        xs = np.array(xs)
    listm = []
    listl = []
    listh = []
    if sigma == "a":
        sigfrac = 0.683    ### 1-sigma
        median = quantile(xs, 0.5)
        lval = quantile(xs, (1 - sigfrac) / 2.0)
        hval = quantile(xs, (1 + sigfrac) / 2.0)
        listm.append(median)
        listl.append( lval-median)
        listh.append(hval-median)

    elif sigma =="b":
        sigfrac = 0.9545    ### 2-sigma
        median = quantile(xs, 0.5)
        lval = quantile(xs, (1 - sigfrac) / 2.0)
        hval = quantile(xs, (1 + sigfrac) / 2.0)
        listm.append(median)
        listl.append( lval-median)
        listh.append(hval-median)
    elif sigma == "c":
        sigfrac = 0.9973     ### 3-sigma
        median = quantile(xs, 0.5)
        lval = quantile(xs, (1 - sigfrac) / 2.0)
        hval = quantile(xs, (1 + sigfrac) / 2.0)
        listm.append(median)
        listl.append( lval-median)
        listh.append(hval-median)

    else:
        print("Invalid sigma number")
    return (listm, listl, listh) 

print("The 1 sigma error of the parameter b is : ",  get_error_estimates123(flatchains_qla_LCDM[:, 0], sigma="a"))
print("The 1 sigma error of the parameter Mcal is : ",  get_error_estimates123(flatchains_qla_LCDM[:, 1], sigma="a"))



### USE CHAIN CONSUMER TO GENERATE PLOTS ###

params = ["b",r"${\cal{M}}$"]
cc = ChainConsumer()
c = cc.add_chain(flatchains_qla_LCDM[:,:], parameters=params)
truth =  [BF_aLCDM[0], BF_aLCDM[1]]
c.configure(kde= True, max_ticks=7, summary = False, shade_alpha=0.9, tick_font_size=11, label_font_size=15, sigmas=[1, 2], linewidths=1.2, colors="#673AB7", sigma2d = False, shade =True, flip = False)
fig = c.plotter.plot(figsize=2.0, extents=[[0, 16], [23.75, 23.875]], display=True, truth = truth)

