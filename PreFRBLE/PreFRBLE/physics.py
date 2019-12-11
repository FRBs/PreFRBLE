import numpy as np
import yt


## physical constraints

RM_min = 1 # rad m^-2  ## minimal RM measureable by telescopes, is limited by precision of forground removel of Milky Way and Ionosphere
tau_min = 0.01 # ms    ## minimal tau measureable by telescopes, chosen to be smallest value available in FRBcat. However, depends on telescope, 1 ms for CHIME and ASKAP
tau_max = 50.0 # ms    ## maximal reasonable tau measured by telescopes, chosen to be biggest value observed so far (1906.11305). However, depends on telescope

## considered redshift bins
redshift_bins = np.arange( 0.1,6.1,0.1)
redshift_range = np.arange( 0.0,6.1,0.1)

measures = [ 'DM', 'RM', 'SM', 'tau' ]

units = {
    'DM'       :r"pc cm$^{-3}$",
    'RM'       :r"rad m$^{-2}$",
    'SM'       :r"kpc m$^{-20/3}$",
    'tau'      :"ms",
    'z'        :r"z",
    'redshift' :r"1+z",
}

label_measure = {
    'DM'       : 'DM',
    'RM'       : '|RM|',
    'SM'       : 'SM',
    'tau'      : r"$\tau$",
    'z'        :r"z",
    'redshift' :r"1+z",    
}


scale_factor_exponent = { ## used to redshift results of local
    'DM' : 1,
    'RM' : 2,
    'SM' : 2,
    'tau': 3.4
}



## physical constants                                                                                                          
omega_baryon       = 0.048
omega_CDM          = 0.259
omega_matter       = 0.307
omega_lambda       = 0.693
omega_curvature    = 0.0
hubble_constant    = 0.71
 
from yt.units import speed_of_light_cgs as speed_of_light

## cosmic functions
co = yt.utilities.cosmology.Cosmology( hubble_constant=hubble_constant, omega_matter=omega_matter, omega_lambda=omega_lambda, omega_curvature=omega_curvature )
comoving_radial_distance = lambda z0, z1: co.comoving_radial_distance(z0,z1).in_units('Gpc').value

## physics



def AngularDiameterDistance(z_o=0., z_s=1.):
    ### compute angular diameter distance as measured for
    ### z_o : redshift of observer
    ### z_s : redshift of source

    ## make sure both are arrays
    redshift_observer = z_o if type(z_o) is np.ndarray else np.array([z_o])
    redshift_source = z_s if type(z_s) is np.ndarray else np.array([z_s])

    ## if one is longer than the other, assuming that the other was a single number, repeat accordingly
    if redshift_observer.size > redshift_source.size:
        redshift_source = redshift_source.repeat( redshift_observer.size )
    if redshift_observer.size < redshift_source.size:
        redshift_observer = redshift_observer.repeat( redshift_source.size )

    return np.array([ ( comoving_radial_distance(0,z2) - comoving_radial_distance(0,z1) )/(1+z2) for z1, z2 in zip( redshift_observer, redshift_source )])



''' old and ugly
    if type(z_o) is not np.ndarray:
        if type(z_s) is not np.ndarray:
            return ( comoving_radial_distance(0,z_s) - comoving_radial_distance(0,z_o) )/(1+z_s)
        else:
            return np.array([ ( comoving_radial_distance(0,z) - comoving_radial_distance(0,z_o) )/(1+z) for z in z_s.flat])
    else:
        if type(z_s) is not np.ndarray:
            return np.array([ ( comoving_radial_distance(0,z_s) - comoving_radial_distance(0,z) )/(1+z_s) for z in z_o.flat])         
        else:
            return np.array([ ( comoving_radial_distance(0,z2) - comoving_radial_distance(0,z1) )/(1+z2) for z1, z2 in zip( z_o.flat, z_s.flat )])
'''


def Deff( z_s=np.array(1.0), ## source redshift
         z_L=np.array(0.5)   ## redshift of lensing material
        ):
    ### compute ratio of angular diameter distances of lense at redshift z_L for source at redshift z_s
    D_L = AngularDiameterDistance( 0, z_L )
    D_S = AngularDiameterDistance( 0, z_s )
    D_LS = AngularDiameterDistance( z_L, z_s )   
    return D_L * D_LS / D_S  ## below Eq. 15 in Macquart & Koay 2013


def ScatteringTime( SM=None,  ## kpc m^-20/3, effective SM in the observer frame
                   redshift=0.0, ## of the scattering region, i. e. of effective lense distance
                   D_eff = 1., # Gpc, effective lense distance
                   lambda_0 = 0.23, # m, wavelength
                  ):
    ### computes scattering time in ms of FRB observed at wavelength lambda_0, Marcquart & Koay 2013 Eq.16 b
    return 1.8e5 * lambda_0**4.4 / (1+redshift) * D_eff * SM**1.2
    
def Freq2Lamb( nu=1. ): # Hz 2 meters
    return speed_of_light.in_units('m/s').value / nu

def Lamb2Freq( l=1. ): # meters 2 Hz
    return speed_of_light.in_units('m/s').value / l

HubbleParameter = lambda z: co.hubble_parameter(z).in_cgs()
def HubbleDistance( z ):
    return (speed_of_light / HubbleParameter(z)).in_units('Mpc').value

def PriorInter( z_s=6.0,   ## source redshift
            r=1., ## Mpc, radius of intervening galaxy
            n=1 , ## Mpc^-3 number density of galaxies
            comoving = False ## indicates whether n is comoving
           ):
    ### compute the prior likelihood of galaxies at redshift z to intersect the LoS, integrand of Macquart & Koay 2013 Eq. 33
    z = redshift_bins[redshift_bins<=z_s]
    if (type(n) is not np.ndarray) or comoving:
        ## for comoving number density, consider cosmic expansion
        n = n * (1+z)**3
    return np.pi* r**2 * n * HubbleDistance(z) / (1+z)

def nInter( z_s=6.0,   ## source redshift
            r=1., ## Mpc, radius of intervening galaxy
            n=1 , ## Mpc^-3 number density of galaxies
            comoving = False ## indicates whether n is comoving
           ):
    ### compute the average number of LoS intersected by a galaxy at redshift z, Macquart & Koay 2013 Eq. 33
    dz = np.diff(redshift_range[redshift_range<=z_s*1.000001]) ## small factor required to find correct bin, don't know why it fails without...
    pi_z = PriorInter( z_s, r=r, n=n, comoving=comoving)
    return  pi_z * dz
    

def NInter( z_s=6.,   ## source redshift
            r=1., ## radius of intervening galaxy
            n=1 , ## number density of galaxies
            comoving = False ## indicates whether n is comoving
           ):
     ### returns total intersection likelihood for source at all redshift bins up to z_s
    return np.cumsum( nInter( z_s, r=r, n=n, comoving=comoving) )




