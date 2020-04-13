import numpy as np
import yt


## considered redshift bins
redshift_bins = np.arange( 0.1,6.1,0.1)
redshift_range = np.arange( 0.0,6.1,0.1)

############################################################################
############################## MEASURES ####################################
############################################################################


measures = [ 'DM', 'RM', 'SM', 'tau' ]

units = {
    'DM'       :r"pc cm$^{-3}$",
    'RM'       :r"rad m$^{-2}$",
    '|RM|'       :r"rad m$^{-2}$",
    'SM'       :r"kpc m$^{-20/3}$",
    'tau'      :"ms",
    'z'        :r"",
    'redshift' :r"1+z",
}

label_measure = { ## labels used in plots
    'DM'       : 'DM',
    'RM'       : '|RM|',
    '|RM|'     : '|RM|',
    'SM'       : 'SM',
    'tau'      : r"$\tau$",
    'z'        :r"z",
    'redshift' :r"1+z",    
}

scale_factor_exponent = { ## exponent of scale factor a=1/(1+z) in redshift dependence of measure
    'DM' : 1,
    'RM' : 2,
    '|RM|' : 2,
    'SM' : 2,
    'tau': 3.4
}

## physical constraints

RM_min = 1 # rad m^-2  ## minimal RM measureable by telescopes, is limited by precision of forground removel of Milky Way and Ionosphere
tau_min = 0.01 # ms    ## minimal tau measureable by telescopes, chosen to be smallest value available in FRBcat. However, depends on telescope, 1 ms for CHIME and ASKAP
tau_max = 50.0 # ms    ## maximal reasonable tau measured by telescopes, chosen to be biggest value observed so far (1906.11305). However, depends on telescope



measure_range = { ## range of values accesible by telescope. should be set individually for each instrument
    'DM'       : (None, None),
    'RM'       : (RM_min, None),
    'tau'      : (tau_min, tau_max)
}





############################################################################
########################## PHYSICAL CONSTANTS ##############################
############################################################################

## cosmological constants                                                                
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


sigma_probability = { ## proability entailed within sigma range
    1: 0.682689492,
    2: 0.954499736,
    3: 0.997300204,
    4: 0.99993666,
    5: 0.999999427
}



############################################################################
########################## PHYSICS FUNCTIONS ###############################
############################################################################



def Jackknife( values, function=np.mean, axis=None ):
    """  
    obtain Jackknife estimator
    
    Parameter
    ---------
    values : array-like
        values to be Jackknifed
    axis : integer
        axis along which to Jackknife
    function : function-type
        function to be executed on the samples to find the value to be estimated
        needs keyword axis if axis is used
        
    Returns
    -------
    mean, std : floats
        average and standard deviation of esimates computed by function
    """
    estimates = values.copy()
    
    #N_samples = values.shape[axis] if axis else values.size
    if axis:
        estimates = np.moveaxis( estimates, axis, 0 )
        for i in range(estimates.shape[0]):
            estimates[i] = function( np.delete( values, i, axis=axis), axis=axis )
        estimates = np.moveaxis( estimates, 0, axis  )
    else:
        for i in range(values.size):
            estimates.flat[i] = function( np.delete( values, i ) )        
    return np.nanmean(estimates, axis=axis), np.nanstd(estimates, axis=axis)



def AngularDiameterDistance(z_o=0., z_s=1.):
    """ compute angular diameter distance between redshift of observer z_o and source z_s """

    ## make sure both are arrays
    redshift_observer = z_o if type(z_o) is np.ndarray else np.array([z_o])
    redshift_source = z_s if type(z_s) is np.ndarray else np.array([z_s])

    ## if one is longer than the other, assuming that the other was a single number, repeat accordingly
    if redshift_observer.size > redshift_source.size:
        redshift_source = redshift_source.repeat( redshift_observer.size )
    if redshift_observer.size < redshift_source.size:
        redshift_observer = redshift_observer.repeat( redshift_source.size )

    return np.array([ ( comoving_radial_distance(0,z2) - comoving_radial_distance(0,z1) )/(1+z2) for z1, z2 in zip( redshift_observer, redshift_source )])



def Deff( z_s=np.array(1.0), ## source redshift
         z_L=np.array(0.5)   ## redshift of lensing material
        ):
    """ compute ratio of angular diameter distances of lense at redshift z_L for source at redshift z_s (see Eq. 15 in Macquart & Koay 2013) """
    D_L = AngularDiameterDistance( 0, z_L )
    D_S = AngularDiameterDistance( 0, z_s )
    D_LS = AngularDiameterDistance( z_L, z_s )   
    return D_L * D_LS / D_S


def ScatteringTime( SM=None,  ## kpc m^-20/3, effective SM in the observer frame
                   redshift=0.0, ## of the scattering region, i. e. of effective lense distance
                   D_eff = 1., # Gpc, effective lense distance
                   lambda_0 = 0.23, # m, wavelength
                  ):
    """ computes scattering time in ms of FRB observed at wavelength lambda_0, Marcquart & Koay 2013 Eq.16 b """
    return 1.8e5 * lambda_0**4.4 / (1+redshift) * D_eff * SM**1.2
    
def Freq2Lamb( nu=1. ):
    """ transform frequency in Hz to wavelength in meters """
    return speed_of_light.in_units('m/s').value / nu

def Lamb2Freq( l=1. ): 
    """ transform wavelength in meters to frequency in Hz """
    return speed_of_light.in_units('m/s').value / l

HubbleParameter = lambda z: co.hubble_parameter(z).in_cgs()
def HubbleDistance( z ):
    return (speed_of_light / HubbleParameter(z)).in_units('Mpc').value

def PriorInter( z_s=6.0, model='Rodrigues18' ):
    """ 
    Compute the likelihood of LoS to be intersected by a galaxy at redshift for all redshifts in redshift_bins (Macquart & Koay 2013 Eq. 33). All units in Mpc.
    Results are given for all redshift_bins <= z_s

    Parameter
    ---------
    z_s : float
       source redshift
    model : string
       mnemonic of the intervening galaxy model

    Return
    ------
    pi_inter : numpy-array
        likelihood of LoS to be intersected by galaxy at corresponding redshift in redshift_bins
        for intersection probability, multiply by size of redshift bin, which is done by nInter

    """

    ## use values only to required redshift
    iz = np.where( redshift_bins == z_s )[0][0] + 1
    z = redshift_bins[:iz]


    if model == 'Macquart13':
        ## values adopted from Marcquart & Koay 2013
        n = 0.02*hubble_constant**3  
        r = 0.01/hubble_constant     

    elif model == 'Rodrigues18':
        ## values correpsonding to settings used in Rodrigues et al. 2018
        
        ## read data written to dat_file
        d = np.genfromtxt(Rodrigues_file_rgal, names=True)

        r = d['r_gal'][:iz] # kpc
        ## use correct r_gal = 2.7 * r_1/2 and units (Mpc)
        r *= 2.7e-3 # Mpc

        n = d['n_gal'][:iz] # Mpc-3, comoving
        comoving=True
        ### n_gal is broken somehow, so for now we use 
        n = 0.089 # Mpc-3

    if (type(n) is not np.ndarray) or comoving:
        ## for constant or comoving number density, consider cosmic expansion
        n = n * (1+z)**3
    return np.pi* r**2 * n * HubbleDistance(z) / (1+z)

def nInter( z=6.0, model='Rodrigues18' ):
    """ 
    compute the average number of LoS intersected by a galaxy in redshift bins (Macquart & Koay 2013 Eq. 33). All units in Mpc.
    Results are given for all redshift_bins <= z_s

    Parameter
    ---------
    z : float
       source redshift
    model : string
       mnemonic of the intervening galaxy model

    Return
    ------
    n_inter : numpy-array
        average amount of LoS intersected by galaxy at corresponding redshift in redshift_bins

    """
    dz = np.diff(redshift_range[redshift_range<=z*1.000001]) ## small factor required to find correct bin, don't know why it fails without...
    pi_z = PriorInter( z_s=z, model=model)
    return  pi_z * dz
    
from PreFRBLE.file_system import Rodrigues_file_rgal
def NInter( z=6., model='Rodrigues18' ):
    """ 
    compute the average number of LoS to redhshift z_s intersected by a galaxy (Macquart & Koay 2013 Eq. 33). All units in Mpc 

    Parameter
    ---------
    z : float
       source redshift
    model : string
       mnemonic of the intervening galaxy model

    Return
    ------
    N_inter : float
        average amount of galaxies intersecting LoS to source at corresponding redshift in redshift_bins

    """

    ## use n and r only to redquired redshift
    iz = np.where( redshift_bins > z )[0][0]

    if model == 'Rodrigues18':
        ## read data written to dat_file
        d = np.genfromtxt(Rodrigues_file_rgal, names=True)

        r = d['r_gal'][:iz] # kpc
        ## use correct r_gal = 2.7 * r_1/2 and units (Mpc)
        r *= 2.7e-3 # Mpc

        n = d['n_gal'][:iz] # Mpc-3
        comoving=True
        ### n_gal is broken somehow, so for now we use 
        n = 0.089 # Mpc-3

    return np.sum( nInter( z_s=z, r=r, n=n, comoving=comoving) )


def LogMeanStd( data, axis=None ):
    """ return logarithmic mean and standard deviation such to easily plot with pyplot.errorbar"""
    mean_log = np.mean( np.log10( data ), axis=axis )
    mean = 10.**mean_log
    std = np.std( np.log10( data ), axis=axis )
    dev = np.array( [ mean - 10.**(mean_log-std), 10.**(mean_log+std) - mean ] )
    if len(dev.shape) == 1:
        dev = dev.reshape( [2,1] )
    return mean, dev


