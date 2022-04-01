import numpy as np
from numpy import *
from astropy import units as u
from scipy.integrate import quad
import math as math
from math import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib import rcParams as rcp
from matplotlib import colors
from matplotlib import rc
plt.rcParams['figure.figsize'] = [9, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
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
rcp['savefig.dpi'] = 300
rcp["figure.dpi"] = 100
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.angles import Angle
import sympy as sp
c_km = (c.to('km/s').value)
######## PANTHEON data   ######################
# import redshifts (cosmological + heliocentric), apparent magnitudes, error of app.magnitudes, systematic errors
######## PANTHEON data   ######################
# import redshifts (cosmological + heliocentric), apparent magnitudes, error of app.magnitudes,systematic errors

# Get the data from the github repository Pantheon of Dan Scolnic #
# https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long_zhel.txt #
data = np.loadtxt('/home/kerky/anaconda3/SN1A_DATA/PANTHEON_DATA/Scolnic_data_updated.txt', usecols=[1,2,4,5])

### list with all the systematics as found in the PANTHEON catalog ###

# get the full systematics from the same repository #
#https://github.com/dscolnic/Pantheon/blob/master/sys_full_long.txt #
sys = np.genfromtxt('/home/kerky/anaconda3/SN1A_DATA/PANTHEON_DATA/systematics.txt', skip_header=1)

### The list sn_names contains all the supernovae names in the PANTHEON catalog ###
sn_names = np.genfromtxt('/home/kerky/anaconda3/SN1A_DATA/PANTHEON_DATA/Scolnic_data_updated.txt', usecols=[0],dtype='str')

z_cmb= np.array((data[:,0])) ## CMB redshift
z_hel = np.array(np.array(data[:,1])) ## heliocentric redshift
mb = np.array(data[:,2]) ## apparent magnitude


### We select the C11 Scattering Model.
names = np.genfromtxt('/home/kerky/anaconda3/SN1A_DATA/PANTHEON_DATA/Ancillary_C11.FITRES.txt',dtype='str', skip_header=67, usecols=[1])

########## SEPARATE THE Pantheon sample into 13 subsamples based on idsurvey ##########
idsurvey1 = np.genfromtxt('/home/kerky/anaconda3/SN1A_DATA/PANTHEON_DATA/Ancillary_C11.FITRES.txt', skip_header=67, usecols=[3], dtype="str").astype(float)
### The idsurvey1= 61, 62, 63, 64, 65, 66 constitute the CfA
### The idsurvey1=61 constitutes the CfA1
### The idsurvey1=62 constitutes the CfA2
### The idsurvey1=65, 66 constitute the CfA4
### The idsurvey1=63, 64 constitute the CfA3

### The idsurvey1=15 constitutes the PS1
### The idsurvey1=4 constitutes the SNLS
### The idsurvey1=100, 101, 106 constitute the HST
### The idsurvey1=1 constitute the SDSS
### The idsurvey1=5 constitutes the CSP 
xx_high = z_cmb[(idsurvey1!=15) & (idsurvey1!=1) & (idsurvey1!=4) & (idsurvey1!=5) & 
 (idsurvey1!=61) & (idsurvey1!=62) & (idsurvey1!= 63) &(idsurvey1!=64) & (idsurvey1!=65) & 
 (idsurvey1!=66)]
print(len(xx_high))
print(np.min(xx_high))
print(np.max(xx_high))
print(np.median(xx_high))

xx_low = z_cmb[(idsurvey1!=15) & (idsurvey1!=1) & (idsurvey1!=4) & (idsurvey1!=5) & (z_cmb<0.7)]
print(len(xx_low))
print(np.min(xx_low))
print(np.max(xx_low))
print(np.median(xx_low))

xx_SDSS = z_cmb[idsurvey1==1]
print(len(xx_SDSS))
print(np.min(xx_SDSS))
print(np.max(xx_SDSS))
print(np.median(xx_SDSS))

xx_SNLS = z_cmb[idsurvey1==4]
print(len(xx_SNLS))
print(np.min(xx_SNLS))
print(np.max(xx_SNLS))
print(np.median(xx_SNLS))

xx_PS1 = z_cmb[idsurvey1==15]
print(len(xx_PS1))
print(np.min(xx_PS1))
print(np.max(xx_PS1))
print(np.median(xx_PS1))

xx_csp = (z_cmb[idsurvey1==5])
print(len(xx_csp))
print(np.min(xx_csp))
print(np.max(xx_csp))
print(np.median(xx_csp))
############ END #############################################

### COORDINATES ###
### Import the full uncorrected data given in https : // github.com/dscolnic/Pantheon/tree/master/data_fitres.
### We select the C11 Scattering Model.
eqcoor0 = np.genfromtxt('/home/kerky/anaconda3/SN1A_DATA/PANTHEON_DATA/Ancillary_C11.FITRES.txt',dtype=None, skip_header=67, usecols=np.arange(2,45))

### Right-Ascension of the SN as retrieved from the file above.
### The last 18 values are all zero !
### The Ra and DEC of each SnIa are given in degrees. 
ras = np.array([eqcoor0[i][32] for i in range(len(eqcoor0))])
### Declination of the SN as retrieved from the file above.
### The last 18 values are all equal to zero!
decs = np.array([eqcoor0[i][33] for i in range(len(eqcoor0))])

### Print the names of the last 18 SN in the list 
print(names[1030:1048])
### Retrieve the 18 missing values of RA and DEC from https://sne.space/
ra18 = np.array([53.15630,189.28829,189.33646, 53.09299, 189.05763, 189.09550, 189.23633, 189.14522, 53.04175, 53.10558, 189.53734, 53.07563, 189.22552, 53.10326, 189.37083, 35.42527, 35.44368, 187.35689])
dec18 = np.array([-27.77956, 62.19116,62.22819, -27.74084, 62.20210, 62.30644, 62.21481,  62.263576, -27.83055, -27.75084, 62.31316, -27.73626, 62.13950, -27.77169, 62.19106, -3.36476, -3.38227, 1.84905 ] )
### Join the 18 last values of the right ascension and declination with the rest 
ras[ras==0] = ra18
print(ras[1030:1048])
decs[decs==0] = dec18
print(decs[1030:1048])
print(len(ras), len(decs))

### Transform the equatorial coordinates to galactic 
skc = SkyCoord(ra = ras*u.degree, dec=decs*u.degree, frame = 'icrs')
lg, b = skc.galactic.l.value, skc.galactic.b.value

############### Mollweide projection ###############################
### This corresponds to the Figure 1 of our paper 
###  The distribution of 1048 Pantheon SN in the galactic coordinate system
def plot_mwd(ra,dec, org = 0,label = 'label', color = 'black'):
    '''
    RA, Dec are arrays of the same length RA takes values in [0,360), Dec in [-90,90],
    which represent angles in degrees.
    '''
    from matplotlib import colors
    x=np.remainder(ra+360-org, 360)
    ind=x>180
    x[ind]-=360  # scale conversion to [-180, 180]
    #x=-x
    tick_labels=np.array(['210 \xb0'
, '240 \xb0', '270 \xb0', '300 \xb0' ,'330 \xb0', '0 \xb0', '30 \xb0', '60 \xb0', '90 \xb0', '120 \xb0', 
'150 \xb0',])
    #tick_labels=np.remainder(tick_labels+360,360)
    fig = plt.figure(figsize=(14,7))
    vmin = np.min(z_cmb)
    vmax = 2.27
    ax = fig.add_subplot(111,projection="mollweide")
    #cmap = plt.get_cmap('jet', 9)
    cmap = colors.ListedColormap(["blue", "aqua", "green", "greenyellow", "yellow", "gold", "orange", "red"])
    bounds = [0, 0.2, 0.4, 0.6, 0.8, 1, 2.3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax.scatter(np.radians(x),np.radians(dec), marker="o", label = label,c = z_cmb, cmap = cmap, norm = norm, s=6)
    cbar = fig.colorbar(im, ax=ax, orientation = "horizontal")
    cbar.ax.tick_params(labelsize=9)    
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.title.set_fontsize(12)
    ax.set_xlabel("l")
    ax.set_ylabel("b")
    ax.yaxis.label.set_fontsize(10)
    ax.grid(True, which='both')
    plt.show()
plot_mwd(lg, b)
########### END of Mollweide #################################

##################################################################################
### Redshift distribution of the Pantheon sample wrt the different subsamples ####
##################################################################################

### This corresponds to Figure 2 of our paper ### 
xx_lows = z_cmb[(idsurvey1!=15) & (idsurvey1!=1) & (idsurvey1!=4) & (z_cmb<0.7)] 
fig, ax = plt.subplots()
#plt.figure(figsize=(8,6))
plt.hist(xx_lows, bins=20, alpha=0.9, label="low-$z$")
plt.hist(xx_PS1, bins=20, alpha=0.5, label="$PS1$")
plt.hist(xx_SDSS, bins=20, alpha=0.5, label="$SDSS$")
plt.hist(xx_SNLS, bins=20, alpha=0.7, label="$SNLS$")
plt.hist(xx_high, bins=20, alpha=0.5, label="high-$z$")
plt.xlabel("redshift (z)", size=14)
plt.ylabel("Number of SnIa", size=14)
ax.set_xticks([0.2, 0.5, 1, 1.5, 2, 2.3])
plt.xlim(0.007, 2.29)
plt.legend(loc='upper right')

