'''
Physical quantities and functions

'''


import yt
from yt.utilities.cosmology import Cosmology
from yt.units import speed_of_light_cgs as speed_of_light

## units of physical quantities
units = {
    'DM'       :r"pc cm$^{-3}$",
    'RM'       :r"rad m$^{-2}$",
    'SM'       :r"kpc m$^{-20/3}",
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
    'redshift' : 0
}


## conversion factors
Mpc2cm = yt.units.Mpc.in_cgs().value*1 ## *1 to get value instead of 0D-array
kpc2cm = yt.units.kpc.in_cgs().value*1 
h=0.71
Mpch2cm = Mpc2cm/h
kpch2cm = kpc2cm/h

## physical constants
OmegaBaryon       = 0.04
OmegaCDM          = 0.23
OmegaMatter       = 0.27
OmegaLambda       = 0.73
OmegaCurvature    = 0.0

critical_density = 9.47e-30 # g/cm**3
electron_mass = 9.11e-28  # g
proton_electron_mass_ratio = 1836.
proton_mass = 1.67e-24 # g


## cosmic functions
co = Cosmology( hubble_constant=h, omega_matter=OmegaMatter, omega_lambda=OmegaLambda, omega_curvature=OmegaCurvature )

comoving_radial_distance = co.comoving_radial_distance  ## distance z0 to z1

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

def DispersionMeasure( density, distance, redshift, B_LoS=None ):
    ## compute dispersion measure DM
    ### density in g/cm^3
    ### distance in kpc
    ###  B_LoS in G
    ## electron density = gas density / ( proton mass * electron weight )
    DM = density / (proton_mass * 1.16) * distance / (1 + redshift ) / kpc2cm * 1e3 # cgs to pc / cm**3
    if B_LoS is None:
        return DM
    ## if B_LoS is given, also compute the rotation measure
    RM = 0.81 * B_LoS * DM / (1+redshift) * 1e6 # rad / m^2
    return DM, RM

def DispersionMeasureIntegral( density, distance, redshift, B_LoS=None, axis=None ):
    ## compute disperion measure integral
    if B_LoS is None:
        DM = DispersionMeasure( density, distance, redshift )
        return np.sum( DM, axis=axis )
    ## if B_LoS is given, also compute the rotation measure integral
    DM, RM = DispersionMeasure( density, distance, redshift, B_LoS )
    return np.sum( DM, axis=axis ), np.sum( RM, axis=axis )





