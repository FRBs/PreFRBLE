'''
Procedures to 
 - compute the probablity function for "near" and "chopped" LoS
 - read probability functions from file (!!! remove, double ???)

'''


from Models import *
from Parameters import *
from Skymaps import GetSkymap
import numpy as np, h5py as h5


def MakeNearProbabilityFunction( model=model, value='DM', nside=nside, bins=30, absolute=False, range=None ):
    ## reads near skymaps, computes their probability functions and writes to probability_file_IGM

    ## for each redshift
    for z in redshift_skymaps_near[1:]:
    ##   read skymap
        sky = GetSkymap( z, typ='near', model=model, value=value, nside=nside )
        if absolute:
            sky = np.abs( sky )
        if value=='DM' or absolute:
            ## for definitely positive values, use logarithmic bins
            log = True
        else:
            log = False

    ##   compute probability function, which is the density histogram of the full sky, normalized to 1 = int P dx
        P, x = histogram( sky, density=True, bins=bins, range=range, log=log )

    ##   write to file
        keys = [ KeyProbability( z, model, 'near', nside, '|%s|' % value if absolute else value, which ) for which in ['P', 'x'] ]
        Write2h5( probability_file_IGM, [P,x], keys )

    

def MakeDMRMRayProbabilityFunction( nbins, x_range, bunch=128, typ='DM', model=model, absolute=True ):
    ## compute likelihood function of LoS observables of "chopped" rays at high redshift in DMRMrays_file

    ## empty array for final results
    histograms = np.zeros([len(redshift_skymaps[1:]),nbins])
    i_ray, n_iter, n_rays = 0, 0, 0

    with h5.File( DMRMrays_file ) as f:
        ## find the number of rays to be considered
        i_ray_max = max( np.array( f["/%s/chopped" % model].keys() ).astype('i') )

        
        while i_ray < i_ray_max:
            rays = []
            ## read the data of a bunch of rays
            for i in range(bunch):
##                i_ray += 1   ## moved to later in order to not skip i=0
##                if i_ray > i_ray_max:
##                    break
                key = '/'.join( [ model, 'chopped',  str(i_ray), typ] )
                rays.append( f[key].value )
                n_rays += 1 
                i_ray += 1     ##
                if i_ray > i_ray_max:
                    break
            if len(rays) == 0:
                continue
####            rays = np.cumsum( rays, axis=1 )
            if absolute:
                rays = np.abs(rays)
            ## for each redshift, compute the likelihood function of the bunch and add to full result, weighted by number of rays in current bunch
            histograms += float(len(rays))*np.array( [ histogram( rays[:,i], bins=nbins, range=x_range, density=True, log=True if typ=='DM' or absolute else False )[0] for i in range(len(redshift_skymaps[1:]) ) ] )

            n_iter += 1
            print '%.0f percent' % ( 100*float(i_ray)/i_ray_max ), 
    print n_rays
    ## renormalize to 1
    histograms /= n_rays

    ## range of the logarithmic histogram bins 
    x = 10.**np.linspace( *np.log10(x_range), num=nbins+1 )
    
    ## write to file
    for i_z, z in enumerate( redshift_skymaps[1:] ):
        keys = [ KeyProbability( z, model, 'far', nside, '|%s|' % typ if absolute else typ, which ) for which in [ 'P', 'x' ] ]
        Write2h5( probability_file_IGM, [histograms[i_z], x], keys )
    return histograms


def GetProbability( z, model=model, typ='near', nside=nside, value='DM', absolute=False ):
    ## read likelihood function from probability_file_IGM
    keys = [ KeyProbability( z, model, typ, nside, '|%s|' % value if absolute else value, which ) for which in [ 'P', 'x' ] ]
    return tuple([ h5.File( probability_file_IGM )[key].value for key in keys ])

