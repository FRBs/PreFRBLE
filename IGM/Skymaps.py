'''
compute skymaps for the results of "near" LoS
'''

from Physics import *
from Parameters import *
from Models import *
from Rays import *

from tqdm import trange




def MakeNearSkymaps( models=[model] ):
    ## computes full-sky maps of observables from raw ray data in rays_file created by Rays.py/MakeNearRay 
    ## writes them to skymap_file
    ## creates RM for several models by renormalizing the magnetic field based on B~rho relations provided in relation_file

    ## create empy array for results
    skymaps = np.zeros( [ 2+len(models) , N_skymaps_near, npix ] )

    ## define relation function from bins given in relation_file
    f = [ np.genfromtxt( relation_file % m, names=True ) for m in models ]
    def renorm( i, rho ):
        ix = np.array( [ np.where( f[i]['density'] >= rhoi )[0][0] for rhoi in rho ] )
        return f[i]['Renorm'][ix]
    
    ## for each ray
    for ipix in trange( npix ):
    ##     read ray data
        data = GetNearRayData( ipix )      
    ##     compute observables
        DM = DispersionMeasure( density=data['Density'], distance=data['dl'], redshift=data['redshift'] )
        RM = RotationMeasure( DM=DM, B_LoS=data['B_LoS'] )
        SN = ScatteringMeasure( density=data['Density'], distance=data['dl'], redshift=data['redshift'], outer_scale=outer_scale_0_IGM )
    ##     compute RM for magnetic field models
        RMs = []
        for im, m in enumerate( models ):
    ##      apply renormalization factor
            RMs.append( RM * renorm( im, data['Density'] / ( critical_density*omega_baryon*(1+data['redshift'])**3 ) ) ) ## density in terms of average (baryonic) density

    ##     collect to corresponding skymap
        for i, (z0,z1) in enumerate( zip( redshift_skymaps_near, redshift_skymaps_near[1:] ) ):
            izs = np.where( ( z0 < data['redshift'] ) * ( data['redshift'] <= z1 ) )[0]
            skymaps[0,i, ipix] = np.sum( DM[izs] )
            skymaps[1,i, ipix] = np.sum( SM[izs] )
            for im, m in enumerate( models ):
                skymaps[im+2,i, ipix] = np.sum( RMs[im][izs] )
    
    ## add results from lower redshift
    skymaps = np.cumsum( skymaps, axis=1 )

    ## write to file
    Write2h5( skymap_file, skymaps[0], [ KeySkymap( z, model, 'near', nside, 'DM' ) for z in redshift_skymaps_near[1:] ] )
    Write2h5( skymap_file, skymaps[1], [ KeySkymap( z, model, 'near', nside, 'SM' ) for z in redshift_skymaps_near[1:] ] )
    for im, m in enumerate( models ):
        Write2h5( skymap_file, skymaps[2+im], [ KeySkymap( z, m, 'near', nside, 'RM' ) for z in redshift_skymaps_near[1:] ] )
    

    


def GetSkymap( z, model=model, typ='near', nside=nside, measure='DM' ):
    ## read data of indicated skymap from skymap_file
    key = KeySkymap( z, model, typ, nside, measure )
    return h5.File( skymap_file )[key].value
