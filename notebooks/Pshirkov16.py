"""
Calculate Rotation Measures and Dispersion Measures from Monte-Carlo
simulation of cosmic structure

References:
-----------
Pshirkov et al. (2015)
https://arxiv.org/abs/1504.06546v2

input for class:

B0: magnetic field in muG at reference density n0
l0: correlation of B-field length in Mpc

Example:

from DM_Hackstein import Sightline
import numpy as np
import matplotlib.pyplot as plt
a = Sightline(1.e-3,10) # assuming 1 nG at n=1.8e-7 and correlation length of 10 Mpc
z,DM,RM = a.MC_DM_RM(100,0.01,3,40)    # for 100 sight lines in each of the 50 bins in redshift

plt.plot(z,DM)
plt.yscale('log')
plt.show()

Hamburg 2020

for name, module in sys.modules.items():
    if 'RM' in name:
        importlib.reload(module)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.cosmology import Planck13, z_at_value
import astropy.units as u

class Sightline:

    def __init__(self,B0,l0, z_max):
        self.B0 = B0           # muGauss at number density
        self.l0 = l0           # correlation of B-field length in Mpc
        self.refdens = 1.8e-7

        self.zarray = self.CreateArrays( z_max )[0]

    def MC_DM_RM(self, nmax,zmin,zmax,numbins):
        """
        nmax = number of random draws
        zmin = minimum redshift
        zmax = maximum redshift
        numbins = number of logarithmic bins in redshift
        """
        
        DMbins = np.zeros(numbins)
        RMbins = np.zeros(numbins)
        zbins = np.logspace(np.log10(zmin),np.log10(zmax), numbins)
        for i in range(numbins):
            DMbins[i]=np.mean(self.DM(nmax,zbins[i]))
            RMbins[i]=np.mean(self.RM(nmax,zbins[i]))
            print(i*100/numbins,' %')
        return zbins,DMbins, RMbins
        
    def DM(self,nmax,zmax):
        """
        nmax: number of random sight lines
        Returns: DM in units of pc cm^-3
        """
        DM_nmax = np.zeros(nmax)

        zfar = self.zarray[ self.zarray <= zmax ]

        for i in range(nmax):
            zfar,d1 = self.CreateArrays(zmax, zarray=zfar)
            DM_nmax[i] = self.DispersionMeasure(d1,zfar)[-1]
        return DM_nmax

    def RM(self,nmax,zmax):
        """
        nmax: number of random sight lines
        Returns: RM
        """
        RM_nmax = np.zeros(nmax)

        zfar = self.zarray[ self.zarray <= zmax ]  ## use global zarray computed until required redshift

        for i in range(nmax):
            zfar,d1 = self.CreateArrays(zmax, zarray=zfar)
            RM_nmax[i] = self.RotationMeasure(d1,zfar)[-1]
        return RM_nmax

    def JeansLength(self,z):
        """
        Computes Jeans length in unitless Delta z
        lambda_J=2.3 Mpc (1+z)^-3/2 Pshirkov 2016
        """
        dist=Planck13.lookback_distance(z).value # in Mpc
        jeanslength=2.3*(1+z)**(-1.5)              # in Mpc
        z1=z_at_value(Planck13.lookback_distance,(dist+jeanslength)*u.Mpc)
        return z1-z
        
    def NumberDensity(self,z):
        d=Planck13.critical_density(z).value
        return d/1.67e-24 # in cm^-3

    def CreateArrays(self,zmax, zarray=None):
        """
        returns number density and redshift (out to zmax) arrays
        density is sampled from log-normal distribution including correlation
        redshift array is incremented by Jeans length

        Input:  
                zmax - maximum redshift
                zarray - (optional) supply the repeatedly calculated zarray for multiple LoS to allow for faster computation
        """
        if zarray is None:
            zarray=np.array([0])            # this will contain the redshift
        d1=self.NumberDensity(zarray[0])
        densarray=np.array([d1])  # this will contain the number density
        innercutoff=0.2
        z = 0.
        if len(zarray) == 1:
            while z<zmax:
                z=zarray[-1]+self.JeansLength(zarray[-1])
                zarray=np.append(zarray,z)
                sigma = 0.08+5.37/(1+z)-4.21/(1+z)**2+1.44/(1+z)**3
                mu = -0.5*sigma**2
                d = 1.8e-7*np.random.lognormal(mu,sigma,1)*(1+z)**3
                densarray=np.append(densarray,d)
        else:
            sigma = 0.08+5.37/(1+zarray)-4.21/(1+zarray)**2+1.44/(1+zarray)**3
            mu = -0.5*sigma**2
            for z, m, s in zip( zarray[1:], mu[1:], sigma[1:]):
                d = 1.8e-7*np.random.lognormal(m,s,1)*(1+z)**3
                densarray=np.append(densarray,d)

        return zarray, densarray

    def DispersionMeasure(self,densarray,zarray):
        """
        Computes DM on input density array and redshift array.
        """
        properdistancearray=Planck13.lookback_distance(zarray).value
        properdistancearray=np.array(properdistancearray)

        # properdistance is measured in Mpc
        # units of DM is pc cm^-3 -> factor of 1e6 in front of integral

        DM=1.e6*np.cumsum(densarray[:-1]*np.diff(properdistancearray)/(1+zarray[:-1]))

        return DM

    def RotationMeasure(self,densarray,zarray):
        """
        Computes RM on input density array and redshift array.
        Assumes that magnetic field scales as density^2/3.
        Draw magnetic field direction from uniform distribution

        """
        properdistancearray=Planck13.lookback_distance(zarray).value
        properdistancearray=np.array(properdistancearray)

        # add random flip after multiple self.l0 - there must be a faster way   
        ### this is not the bottle neck, constructing the densarray and zarray is. 
        ### computing these arrays takes up to > 100 seconds while this only takes 0.2 seconds
        ### however, the version below should be faster
        '''
        magfield = np.ones(len(properdistancearray))
        ampl = self.B0
        counter = 1.
        refdens = 1.8e-7
        for i in range( len(properdistancearray)):
#            if (properdistancearray[i]>counter*self.l0*(densarray[i]/refdens)**-1):
#            if (properdistancearray[i]>counter*self.l0):
            if (properdistancearray[i]>counter*self.l0*2.3*(1+zarray[i])**(-1.5)):
                ampl = self.B0*np.cos(np.random.uniform(0,2*np.pi,1))
                counter = counter+1
            magfield[i] = ampl
        magarray = densarray**(2./3.)*magfield/self.refdens**(2./3.)

        '''
        ## first initiate magnetic field array with maximum possible value
        magarray = self.B0 * ( densarray / self.refdens )**(2./3.) ## refdens has to be in proper coordinates, as is densarray, thus has to be rescaled by (1+z)**3. This gives the comoving magnetic field, which has to be rescaled by (1+z)**2 to get the proper field strength
        ## find the flip number for all steps
        flip = ( properdistancearray // ( self.l0*2.3*(1+zarray)**(-1.5) ) ).astype('i')
        ##  compute B|| for all flips
        r = np.cos(np.random.uniform(0,2*np.pi,flip[-1]+1))
        magarray *= r[flip]

        # In cumulative sum, direction is reversed to integrate toward observer
        # 812e3 because properdistance is measured in Mpc instead of kpc
        RM=812.e3*np.cumsum(densarray[-2::-1]*magarray[-2::-1]*np.diff(properdistancearray[::-1])/(1+zarray[-2::-1])**2)
        #print RM[-1]

        return RM
