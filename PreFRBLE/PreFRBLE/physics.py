import matplotlib.pyplot as plt
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

RM_min = 1 # rad m^-2  ## minimal RM measureable by telescopes, is limited by precision of forground removal of Milky Way and Ionosphere
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


def SampleProbability(  x=(1,10), y=(1,10), z=(2,11), resolution_estimate=10, log=False, plot=False  ):
    """
    Given two random variables within bin of ranges x and y, return the probability for their sum to fall in bin with range z
    
    Parameters
    ----------
    x, y : 2-tuple or similar
        ranges of two random variables, y can be negative. Have to obey x[0] < x[1] and y[0] < y[1]
    z : 2-tuple or similar
        range of possible sum of x and y. z[0] < z[1]
    resolution_estimate : integer
        number of sub-bins used to compute the integral that delivers the intersection probability
    log : boolean
        indecate whether x, y and z are log-scaled.
        this assumes that their sampled distribution is log-uniform instead of uniform
    plot : boolean
        if True: plot graph that visualizes integral for probability
    
    """
    if y[0] >= 0:
        ## for the simple case of both x and y positive
        if z[1] <= x[0]+y[0] or z[0] >= x[-1]+y[-1]:
            ## in case probed range is impossible to hit
            return 0.
    else:  ## if y negative
        range_diff = [ x[0]+y[1], x[1]+y[0] ]  ## difference of maximum and minimum range
        if z[0] >= np.max( np.abs( range_diff ) ):
            return 0
#        elif (x[0]>y[1] or x[1]<y[0]) and z[1] < np.min( np.abs( range_diff ) ):
#            return 0

    
    ## support of first variable
    if log:
        x_range = np.linspace(*np.log10(x), num=resolution_estimate+1) ## log10
    else:
        x_range = np.linspace(*x, num=resolution_estimate+1)
    dx = np.diff(x_range)
    x_center = x_range[:-1]+dx/2
    if log:
        x_center = 10.**x_center  

    ya = np.abs(y)  ## need absolute y several times


    ## minimum and maximum value of y to fit in range
    if y[1] > 0 or True:  ## simple case of constructive contribution
        y_max = z[1] - x_center
        y_min = z[0] - x_center
    else: ## for the case of deconstructive contribution, consider the absolute (also negative results are valid)
        y_min = np.zeros_like( x_center )
        y_max = np.zeros_like( x_center )

        ## find, where result is positive or negative, i. e. x > y or x < y
        x_lo = x_center < ya[0]  ## all combinations are negative
        x_hi = x_center > ya[1] ## all combinations are positive

        y_min[x_lo] = z[0] + x_center[x_lo]
        y_min[x_hi] = x_center[x_hi] - z[1]

        y_max[x_lo] = z[1] + x_center[x_lo]
        y_max[x_hi] = x_center[x_hi] - z[0]

        ## for those bins combinations that contain 0, assume all contribution ~0, thus no chance to hit bin
        y_max[~(x_lo+x_hi)] = y_min[~(x_lo+x_hi)] = np.mean(ya) ### place both mind and max at same value somewhere in the center results in 0 contribution from these bins
    
    ## where exceeds parameter space, set to corresponding border
    y_max[y_max > ya[1]] = ya[1]
    y_max[y_max < ya[0]] = ya[0]

    y_min[y_min < ya[0]] = ya[0]
    y_min[y_min > ya[1]] = ya[1]
 
    if log:
        y_min, y_max = np.sort(np.log10(np.abs([y_min, y_max])), axis=0)
    
    ## total volume of possible combinations of x and y
        V_tot = np.diff(np.log10(x)) * np.abs(np.diff(np.log10(ya)))   ### works for positive and negative y
    else:
        V_tot = np.diff(x)*np.diff(y)
    
    
    ## probablity = integral over maximal - minimal possible contribution, i. e. volume of fitting combintations / volume of possible combinations
    prob = np.sum( (y_max-y_min)*dx ) /V_tot


    if plot:
        plt.plot(np.log10(x_center) if log else x_center,y_min, ls=':', lw=3)
        plt.plot(np.log10(x_center) if log else x_center,y_max,ls='--',lw=3)

        if log:
            plt.hlines( np.log10(ya), *np.log10(x) )
            plt.vlines( np.log10(x), *np.log10(ya) )
            plt.xlabel(r"log$_{10}(x)$")
            plt.ylabel(r"log$_{10}(y)$")
        else:            
            plt.hlines( y, *x )
            plt.vlines( x, *y )
            plt.xlabel('x')
            plt.ylabel('y')
    return prob




def AngularDiameterDistance(z_o=0., z_s=1.):
    """ compute angular diameter distance in Gpc between redshift of observer z_o and source z_s """

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
    """ compute ratio of angular diameter distances in Gpc of lense at redshift z_L for source at redshift z_s (see Eq. 15 in Macquart & Koay 2013) """
    D_L = AngularDiameterDistance( 0, z_L )
    D_S = AngularDiameterDistance( 0, z_s )
    D_LS = AngularDiameterDistance( z_L, z_s )   
    return D_L * D_LS / D_S


def ScatteringTime( SM=None, redshift=0.0, D_eff = 1., lambda_0 = 0.23 ):
    """ 
    compute scattering time in ms of FRB observed at wavelength lambda_0 (Marcquart & Koay 2013 Eq.16 b)
    
    Parameter
    ---------
    SM : float or array-like
        effective scattering measure in the observer frame in kpc m^-20/3
    redshfit: float or array-like
        redshift of lensing material
    D_eff : float or array-like
        effective lense distance in Gpc
    lambda : float
        wavelength in m

    Return
    ------
    tau : float or array-like
        scattering time in ms

    """
    return 1.8e8 * lambda_0**4.4 / (1+redshift) * D_eff * SM**1.2
    
def Freq2Lamb( nu=1. ):
    """ transform frequency in Hz to wavelength in meters """
    return speed_of_light.in_units('m/s').value / nu

def Lamb2Freq( l=1. ): 
    """ transform wavelength in meters to frequency in Hz """
    return speed_of_light.in_units('m/s').value / l

HubbleParameter = lambda z: co.hubble_parameter(z).in_cgs()
def HubbleDistance( z ):
    return (speed_of_light / HubbleParameter(z)).in_units('Mpc').value

def PriorInter( redshift=6.0, model='Rodrigues18', r_gal=None, n_gal=None, n_comoving=False ):
    """ 
    Compute the likelihood of LoS to be intersected by a galaxy at redshift for all redshifts in redshift_bins (Macquart & Koay 2013 Eq. 33). All units in Mpc.
    Results are given for all redshift_bins <= redshift

    Parameter
    ---------
    redshift : float
       source redshift
    model : string
       mnemonic of the intervening galaxy model
    r_gal : float or array-like, optional
       instead of a model, pass galaxy size directly
    n_gal : float or array-like, optional
       instead of a model, pass galaxy number density directly
    n_comoving : boolean
       if true, passed n_gal is comoving
    Return
    ------
    pi_inter : numpy-array
        likelihood of LoS to be intersected by galaxy at corresponding redshift in redshift_bins
        for intersection probability, multiply by size of redshift bin, which is done by nInter

    """

    ## use values only to required redshift 
    iz = redshift_bins <= redshift+0.0001 ### small shift required to find correct bin, don't know why it fails without...
    z = redshift_bins[iz]


    comoving=False
    if model == 'Macquart13':
        ## values adopted from Marcquart & Koay 2013
        n = 0.02*hubble_constant**3  
        r = 0.01/hubble_constant     
        comoving=True

    elif model == 'Rodrigues18':
        ## values correpsonding to settings used in Rodrigues et al. 2018
        
        ## read data written to dat_file
        d = np.genfromtxt(Rodrigues_file_rgal, names=True)

        r = d['r_gal'][iz] # kpc
        ## use correct r_gal = 2.7 * r_1/2 (1% of surface density) and units
        r *= 2.7e-3 # Mpc
        
        n = d['n_gal'][iz] # Mpc-3
        comoving=False
    if r_gal is not None:
        r = r_gal
    if n_gal is not None:
        n = n_gal
        if n_comoving or (type(n_gal) is not np.ndarray):
            comoving = True

    if (type(n) is not np.ndarray) or comoving:
        ## for constant or comoving number density, consider cosmic expansion to use proper density
        n = n * (1+z)**3
    return np.pi* r**2 * n * HubbleDistance(z) / (1+z)

def nInter( redshift=6.0, model='Rodrigues18', **kwargs_PriorInter ):
    """ 
    compute the average number of LoS intersected by a galaxy in redshift_bins (Macquart & Koay 2013 Eq. 33). All units in Mpc.
    Results are given for all redshift_bins <= redshift

    Parameter
    ---------
    redshift : float
       source redshift
    model : string
       mnemonic of the intervening galaxy model

    Return
    ------
    n_inter : numpy-array
        average amount of LoS intersected by galaxy at corresponding redshift in redshift_bins

    """
    dz = np.diff(redshift_range[redshift_range<=redshift+0.0001]) ## small shift required to find correct bin, don't know why it fails without...
    pi_z = PriorInter( redshift=redshift, model=model, **kwargs_PriorInter)
    return  pi_z * dz
    
from PreFRBLE.file_system import Rodrigues_file_rgal
def NInter( redshift=6., model='Rodrigues18', **kwargs_PriorInter ):
    """ 
    compute the average number of LoS to redhshift intersected by a galaxy (Macquart & Koay 2013 Eq. 33). All units in Mpc 

    Parameter
    ---------
    redshift : float
       source redshift
    model : string
       mnemonic of the intervening galaxy model

    Return
    ------
    N_inter : float
        average amount of galaxies intersecting LoS to source at corresponding redshift in redshift_bins

    """

    return np.sum( nInter( redshift=redshift, model=model, **kwargs_PriorInter) )


def LogMeanStd( data, axis=None ):
    """ return logarithmic mean and standard deviation such to easily plot with pyplot.errorbar"""
    mean_log = np.mean( np.log10( data ), axis=axis )
    mean = 10.**mean_log
    std = np.std( np.log10( data ), axis=axis )
    dev = np.array( [ mean - 10.**(mean_log-std), 10.**(mean_log+std) - mean ] )
    if len(dev.shape) == 1:
        dev = dev.reshape( [2,1] )
    return mean, dev


