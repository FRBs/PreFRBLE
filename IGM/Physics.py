'''
Physical quantities and functions

'''


import yt, numpy as np
from yt.utilities.cosmology import Cosmology
from yt.units import speed_of_light_cgs as speed_of_light

## units of physical quantities
units = {
    'DM'       :r"pc cm$^{-3}$",
    'RM'       :r"rad m$^{-2}$",
    'SM'       :r"kpc m$^{-20/3}$",
#    'SM'       :r"10$^{12}$ m$^{-17/3}$",
}


## exponent of cosmic scale factor
comoving_exponent = {
    'Density'  : -3,
    'density'  : -3,
    'Bx'       : -2,
    'By'       : -2,
    'Bz'       : -2,
    'B_LoS'    : -2,
    'dl'       : 1,
    'x'        : 1,
    'y'        : 1,
    'z'        : 1,
    'dx'       : 1,
    'dy'       : 1,
    'dz'       : 1,
    'redshift' : 0,
    'dredshift' : 0,
}


## conversion factors
Mpc2cm = yt.units.Mpc.in_cgs().value*1 ## *1 to get value instead of 0D-array
kpc2cm = yt.units.kpc.in_cgs().value*1 
h=0.71
Mpch2cm = Mpc2cm/h
kpch2cm = kpc2cm/h

## physical constants
omega_baryon       = 0.048
omega_CDM          = 0.259
omega_matter       = 0.307
omega_lambda       = 0.693
omega_curvature    = 0.0

critical_density = 9.47e-30 # g / cm**3
electron_mass = 9.11e-28  # g
proton_electron_mass_ratio = 1836.
proton_mass = 1.67e-24 # g
hubble_constant = 67.11 # H_0 in km / s / Mpc

outer_scale_0_IGM = 1 # pc           ## global outer scale assumed to compute SM, choose other values by SM*L0**(-2/3)

## cosmic functions
co = Cosmology( hubble_constant=h, omega_matter=omega_matter, omega_lambda=omega_lambda, omega_curvature=omega_curvature )

comoving_radial_distance = co.comoving_radial_distance  ## comoving distance z0 to z1
luminosity_distance = co.luminosity_distance            ## proper distance z0 to z1

CriticalDensity = lambda redshift: co.critical_density( redshift ).in_cgs().value*1
critical_density = CriticalDensity(0)

def GasDensity( z ):
    return critical_density*omega_baryon*(1+z)**3


## redshift <-> time
z_from_t = co.z_from_t
t_from_z = co.t_from_z


## Geometry

def GetDirection( x, y, z ):
    ## compute direction of LoS through cells at position x, y, z
    ## direction = end - start
    d = np.array( [ x[-1] - x[0], y[-1] - y[0], z[-1] - z[0] ] )
    return d / np.linalg.norm(d)


def GetBLoS( data, direction=None ):
    ## compute the LoS magnetic field for data of single ray, B_LoS = direction.B
    if direction is None:
        direction = GetDirection( data['x'], data['y'], data['z'] )
    return np.dot( direction, np.array( [ data['Bx'], data['By'], data['Bz'] ] ) )


def ScaleFactor( redshift, redshift0 ):
    ## factor to correct redshift evolution
    ## data is given for redshift of snaphost, rescale to redshift along LoS
    return ( 1 + redshift0 ) / (1+redshift)



## Observables

def DispersionMeasure( density=None, distance=None, redshift=None ):
    ## compute dispersion measure DM
    ### density in g/cm^3
    ### distance in kpc
    ## electron density = gas density / ( proton mass * electron weight )
    return density / (proton_mass * 1.16) * distance / (1 + redshift ) / kpc2cm * 1e3 # pc / cm**3


def RotationMeasure( DM=None, density=None, distance=None, redshift=None, B_LoS=None ):
    ## compute rotation measure RM
    ### DM in pc / cm^3
    ###   or
    ### density in g/cm^3
    ### distance in cm
    ###
    ###  B_LoS in G
    if DM is None:
        DM = DispersionMeasure( density=density, distance=distance, redshift=redshift )
    return 0.81 * B_LoS * DM / (1+redshift) * 1e6 # rad / m^2

def ScatteringMeasure( density=None, distance=None, redshift=None, outer_scale=None, omega_baryon=omega_baryon, overdensity=False ):
    #### !!! returns wrong solutions, use ScatteringMeasure_ZHU instead, replace in the future
    ## compute scattering measure SM (cf. Eq. 9 in Zhu et al. 2018)
    ### plasma density in g/cm^3
    ### distance in cm
    ### outer_scale in pc
    if overdensity:
        gas_density = 1
    else:
        gas_density = GasDensity(redshift)

    return 1.42e-13 * (omega_baryon/0.049)**2 * outer_scale**(-2./3) * ( density/gas_density )**2 * (1+redshift)**2 * distance / kpc2cm # kpc m**(-20/3)


def ScatteringMeasure_ZHU( density=None, redshift=None, dredshift=None, outer_scale=1, omega_baryon=omega_baryon, omega_lambda=omega_lambda, omega_matter=omega_matter, overdensity=False, hubble_constant=hubble_constant/100 ):
    if overdensity:
        gas_density = 1
    else:
        gas_density = GasDensity(redshift)
    if dredshift is None:
        z = redshift[1:]
        dz = np.diff(redshift)
    else:
        z = redshift
        dz = dredshift
    return 1.31e13 / hubble_constant * (omega_baryon/0.049)**2 * outer_scale**(-2./3) * ( density/gas_density )**2 * (1+z)**3 / np.sqrt( omega_lambda + omega_matter*(1+z)**3 ) * dz  /kpc2cm*100

ScatteringMeasure = ScatteringMeasure_ZHU
