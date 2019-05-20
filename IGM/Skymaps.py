'''
compute skymaps for the results of "near" LoS
'''

from Physics import *
from Parameters import *
from Models import *
from Rays import *

from tqdm import trange




def MakeNearSkymaps_MultipleModels( models=[model] ):
    ## creates DM / RM skymaps from raw ray data
    ## collects them to file
    ## creates RM for several models by renormalizing B based on B~rho difference between models
    skymaps = np.zeros( [ 1+len(models) , N_skymaps_near, npix ] )

    ## define relation function from bins given in relation_file
    f = [ np.genfromtxt( relation_file % m, names=True ) for m in models ]
    def renorm( i, rho ):
        ix = np.array( [ np.where( f[i]['density'] >= rhoi )[0][0] for rhoi in rho ] )
        return f[i]['Renorm'][ix]
    
    ## for each ray
    for ipix in trange( npix ):
    ##     read ray data
        data = GetRayData( ipix, 'near' )      
    ##     compute DM RM
        DMRM = np.array( DispersionMeasure( data['Density'], data['dl'], data['redshift'], data['B_LoS'] ) )
    ##     compute RM in models
        RMs = []
        for im, m in enumerate( models ):
    ##      apply renormalization factor
            RMs.append( DMRM[1] * renorm( im, data['Density'] / ( critical_density*OmegaBaryon*(1+data['redshift'])**3 ) ) ) ## density in terms of average (baryonic) density

    ##     collect to corresponding skymap
        for i, (z0,z1) in enumerate( zip( redshift_skymaps_near, redshift_skymaps_near[1:] ) ):
            izs = np.where( ( z0 < data['redshift'] ) * ( data['redshift'] <= z1 ) )[0]
            skymaps[0,i, ipix] = np.sum( DMRM[0,izs] )
            for im, m in enumerate( models ):
                skymaps[im+1,i, ipix] = np.sum( RMs[im][izs] )
    
    ## add results from lower redshift
    skymaps = np.cumsum( skymaps, axis=1 )
    ## write to file
    Write2h5( skymap_file, skymaps[0], [ KeySkymap( z, model, 'near', nside, 'DM' ) for z in redshift_skymaps_near[1:] ] )
    for im, m in enumerate( models ):
        Write2h5( skymap_file, skymaps[1+im], [ KeySkymap( z, m, 'near', nside, 'RM' ) for z in redshift_skymaps_near[1:] ] )
    

    

def MakeNearSkymaps( ):
    ## creates DM / RM skymaps from raw ray data
    ## collects them to file

    skymaps = np.zeros( [ 2, N_skymaps_near, npix ] )
    
    ## for each ray
    for ipix in trange( npix ):
    ##     read ray data
        data = GetRayData( ipix, 'near' )      
    ##     compute DM RM
        DMRM = np.array( DispersionMeasure( data['Density'], data['dl'], data['redshift'], data['B_LoS'] ) )
    ##     collect to corresponding skymap
        for i, (z0,z1) in enumerate( zip( redshift_skymaps_near, redshift_skymaps_near[1:] ) ):
            izs = np.where( ( z0 < data['redshift'] ) * ( data['redshift'] <= z1 ) )[0]
            skymaps[:,i, ipix] = np.sum( DMRM[:,izs], axis=1 )
    
    ## add results from lower redshift
    skymaps = np.cumsum( skymaps, axis=1 )
    ## write to file
    for i, key in enumerate( ['DM', 'RM' ] ):
        Write2h5( skymap_file, skymaps[i], [ KeySkymap( z, model, 'near', nside, key ) for z in redshift_skymaps_near[1:] ] )
#       Write2h5( skymap_file, skymaps[i], [ '/'.join( [ model, 'near', str(nside), key, '%.4f' % z ] ) for z in redshift_skymaps_near[1:] ] )



def GetSkymap( z, model=model, typ='near', nside=nside, value='DM' ):
    key = KeySkymap( z, model, typ, nside, value )
#    key = '/'.join( [ model, typ, str(nside), value, '%.4f' % z ] )
    return h5.File( skymap_file )[key].value
