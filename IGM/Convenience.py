'''
Convenient functions

'''


import numpy as np, yt, os, h5py as h5, sys

from Physics import *
from Models import *



### Probability / Likelihood Functions

def histogram( data, bins=10, range=None, density=None, log=False ):
    ## compute histogram of data array, allows for log-scaled binning and probability density
    if log:
        if range is not None:
            ## use log value of range
            range = np.log10(range)
        ## compute histogram of log data
        h, x = np.histogram( np.log10(data), bins=bins, range=range )
        ## unlog the true value
        x = 10.**x
        h = h.astype('float64')
        if density:
            h = h / ( np.sum( h )*np.diff(x) )
    else:
        h, x = np.histogram( data, bins=bins, range=range, density=density )
    return h, x


def Likelihood2Expectation( P, x, log=True, stddev=True ):
    ## compute the expectation value <x> from the likelihood function P( x[1:] - dx/2 )
    ## this usually comes with a standard deviation

    if log:
        ## for log-scale, use the log-value to compute the mean
        x_log = np.log10(x)
        x_ = x_log[:-1] + np.diff(x_log)/2
    else:
        x_ = x[:-1] + np.diff(x)/2
    ## use the center value of each bin, weighted by the likelihood, to compute mean and standard deviation
    x_mean = np.average( x_, weights=P )
    x_std = np.sqrt( np.sum( P*( x_ - x_mean)**2 ) / np.sum( P ) )
    if log:
        ## unlog the true value
        x_mean = 10.**x_mean
    if stddev:
        return x_mean, x_std
    else:
        return x_mean


### yt convenience functions

def TimeSeries( model_dir ):
    ## loads full data series of model in model_dir into yt
    if os.path.isfile( model_dir+'/DD0004/data0004.cpu0000'):
        ## whether written in cycle dumps, 
        series = model_dir+'/DD????/data????'  
    elif os.path.isfile( model_dir+'/RD0002/RD0002.cpu0000'):
        ## redshift dumps
        series = model_dir+'/RD????/RD????'
    else:
        ## or redshift outputs
        series = model_dir+'/RD????/RedshiftOutput????'
    return yt.load( series )


def RedshiftSnapshots( ts, redshift_max, redshift_max_near, redshift_trans, redshift_accuracy ):
    ## read redshifts of snapshots in yt-TimeSeries ts
    ## return redshift of snapshots, z_snaps, and redshifts that indicate when the snapshots is used, redshift_snapshots
    z_snaps = [ ds.current_redshift for ds in ts ]
    z_snaps.sort()
    ## cut all redshifts >= redshift_max
    z_snaps = np.array(z_snaps)
    redshift_snapshots = list( z_snaps[ np.where( z_snaps < redshift_max ) ] )
    if np.round(redshift_snapshots[0],redshift_accuracy) == 0: 
        # if final snapshot is at z=0, use it from half time since previous snapshot ( add redshift of transition to list )
        redshift_snapshots.append( redshift_trans )
    else:
        ## use final snapshot until z=0 (add 0 to list)
        redshift_snapshots.append( 0. )  
    redshift_snapshots.append( redshift_max_near )

    redshift_snapshots.sort()
    return z_snaps, np.array(redshift_snapshots)

def BoxFractions( ts, domain_width_code, redshift_snapshots ):
    ## get required max_box_fraction for each snapshot
    ##   i. e. minimum number of transitions of simulation volume to reach redshift of next snapshot
    return [ int( np.ceil(
        (comoving_radial_distance(z0,z1)/min(ds.domain_width.in_units('cm'))).d / domain_width_code
    ) )
             for z0, z1, ds in zip( redshift_snapshots[1:], redshift_snapshots[2:], ts[::-1])
    ]



### file management

def Write2h5( filename, datas, keys ):
    ## writes arrays listed in data to h5 file filename using the corresponding keys
    ## overwrites previous entries
    if type(keys) is str:
        sys.exit( 'Write2h5 needs list of datas and keys' )
    with h5.File( filename, 'a' ) as f:
        for data, key in zip( datas, keys ):
            try:
                f.__delitem__( key )
            except:
                pass
            f.create_dataset( key, data=data  )
            
def KeyLikelihood_IGM( z, model, typ, nside, measure, axis ):
    ## get key in likelihood_file_IGM
    return '/'.join( [ model, typ, str(nside), measure, '%.4f' % z, axis ] )

def KeySkymap( z, model, typ, nside, measure ):
    ## get key in skymap_file
    return '/'.join( [ model, typ, str(nside), measure, '%.4f' % z ] )

def KeyNearRay( model, nside, ipix, measure ):
    ## key in rays_file
    return '/'.join( [ model, 'near',str(nside), str(ipix), measure ] )


def FileNearRay( ipix, model='primordial', npix=2**6 ):
    ## name of temporary file for constrained ray
    return root_rays + model + '/ray%i_%i.h5' % (ipix,npix)

            







