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
    # reads near skymaps, computes their probability function and writes to file

    ## for each redshift
    for z in redshift_skymaps_near[1:]:
    ##   read skymap
        sky = GetSkymap( z, typ='near', model=model, value=value, nside=nside )
        if absolute:
            sky = np.abs( sky )
        if value=='DM' or absolute:
            log = True
#            sky = np.log10( np.abs( sky ) )
#            range = None
#            if absolute:
#                range=(1e-3,1e2)
#                range=(-3,2)
#            range = (-3,1)
        else:
            log = False
    ##      make sure P is centered on zero
#            max = np.max( [sky.max(), -sky.min()] )
#            range = ( -max, max )
#            range = ( -0.1, 0.1 )

       

    ##   compute probability function
#        P, x = np.histogram( sky, density=True, bins=bins, range=range )
        P, x = histogram( sky, density=True, bins=bins, range=range, log=log )

#        if value == 'DM' or absolute:
#            x = 10.**x
    ##   write to file
        keys = [ KeyProbability( z, model, 'near', nside, '|%s|' % value if absolute else value, which ) for which in ['P', 'x'] ]
        Write2h5( probability_file, [P,x], keys )
#        key = '/'.join( [ model, 'near', str(nside), value, '%.4f' % z ] )
#        Write2h5( probability_file, P, [key+s for s in ['/value','/range'] ] )

    

def MakeDMRMRayProbabilityFunction( nbins, x_range, bunch=128, typ='DM', model=model, absolute=True ):
    ## compute probability function of typ values
    histograms = np.zeros([len(redshift_skymaps[1:]),nbins])
    i_ray, n_iter, n_rays = 0, 0, 0
    with h5.File( DMRMrays_file ) as f:
        i_ray_max = max( np.array( f["/%s/chopped" % model].keys() ).astype('i') )


#        i_ray_max = 1024

        
        while i_ray < i_ray_max:
            rays = []
            for i in range(bunch):
                i_ray += 1
                if i_ray > i_ray_max:
                    break
                key = '/'.join( [ model, 'chopped',  str(i_ray), typ] )
                rays.append( f[key].value )
                n_rays += 1
            if len(rays) == 0:
                continue
            rays = np.cumsum( rays, axis=1 )
            if absolute:
                rays = np.abs(rays)
#            if typ == 'DM':
#                rays = np.log10( rays )
#            histograms += np.array( [ np.histogram( rays[:,i], bins=nbins, range=x_range, density=True )[0] for i in range(len(redshift_skymaps[1:])) ] )
##            histograms += np.array( [ histogram( rays[:,i], bins=nbins, range=x_range, density=True, log=False )[0] for i in range(len(redshift_skymaps[1:]) ) ] )
            histograms += np.array( [ histogram( rays[:,i], bins=nbins, range=x_range, density=True, log=True if typ=='DM' or absolute else False )[0] for i in range(len(redshift_skymaps[1:]) ) ] )
#            histograms += np.array( [ histogram( rays[:,i], bins=nbins, range=x_range, density=True, log= typ=='DM' )[0] for i in range(len(redshift_skymaps[1:]) ) ] )
            n_iter += 1
            print '%.0f percent' % ( 100*float(i_ray)/i_ray_max ), 
    print n_rays
    histograms /= n_iter

    x = 10.**np.linspace( *np.log10(x_range), num=nbins+1 )
    for i_z, z in enumerate( redshift_skymaps[1:] ):
        keys = [ KeyProbability( z, model, 'far', nside, '|%s|' % typ if absolute else typ, which ) for which in [ 'P', 'x' ] ]
        Write2h5( probability_file, [histograms[i_z], x], keys )
    return histograms


def GetProbability( z, model=model, typ='near', nside=nside, value='DM', absolute=False ):
    keys = [ KeyProbability( z, model, typ, nside, '|%s|' % value if absolute else value, which ) for which in [ 'P', 'x' ] ]
    return tuple([ h5.File( probability_file )[key].value for key in keys ])

