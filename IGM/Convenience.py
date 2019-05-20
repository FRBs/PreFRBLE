'''
Convenient functions

'''


import numpy as np, yt, os, h5py as h5, sys

from Physics import *
from Models import *



### Probability / Likelihood Functions

def histogram( data, bins=10, range=None, density=None, log=False ):
    ## compute histogram of data array, allows for log-scaled binning
    if log:
        if range is None:
            range = tuple( np.log10( [np.min(data), np.max(data) ] ) )
        else:
            range = np.log10(range)
        x = 10.**np.linspace( *range, num=bins+1)
        h = np.histogram( np.log10(data), bins=bins, range=range, density=density )[0]        
        if density:
            h /= np.sum( h )*np.diff(x) 
    else:
        if range is None:
            range = ( np.min(data), np.max(data) )
        x = np.linspace( *range, num=bins+1)
        h = np.histogram( data, bins=bins, range=range, density=density )[0]
    return h, x


def Histogram2Expectation( P, x, log=True, ):
    if log:
        x_log = np.log10(x)
        x_ = x_log[:-1] + np.diff(x_log)/2
    else:
        x_ = x[:-1] + np.diff(x)/2
    x_mean = np.average( x_, weights=P )
    x_std = np.sqrt( np.sum( P*( x_ - x_mean)**2 ) / np.sum( P ) )
    if log:
        x_mean = 10.**x_mean
    return x_mean, x_std


### Physics

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

def GetDirection( x, y, z ):
    d = np.array( [ x[-1] - x[0], y[-1] - y[0], z[-1] - z[0] ] )
    return d / np.linalg.norm(d)


def GetBLoS( data, direction=None ):
    ## compute the LoS magnetic field for data of single ray
    if direction is None:
        direction = GetDirection( data['x'], data['y'], data['z'] )
    return np.dot( direction, np.array( [ data['Bx'], data['By'], data['Bz'] ] ) )



### yt convenience functions

def TimeSeries( model_dir ):
    ## loads full data series of model in model_dir into yt
    if os.path.isfile( model_dir+'/DD0004/data0004.cpu0000'):
        ## whether written in cycle dumps
        series = model_dir+'/DD????/data????'  
    elif os.path.isfile( model_dir+'/RD0002/RD0002.cpu0000'):
        series = model_dir+'/RD????/RD????'
    else:
        ## or redshift outputs
        series = model_dir+'/RD????/RedshiftOutput????'
    return yt.load( series )


def RedshiftSnapshots( ts, redshift_max, redshift_max_near, redshift_trans, redshift_accuracy ):
    ## read redshifts of snapshots in yt-TimeSeries ts
    ## return redshift of snapshots, z_snaps, and redshifts that mark the transition between snapshots
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
    return z_snaps, redshift_snapshots

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
            
def KeyProbability( z, model, typ, nside, value, which ):
    ## get key in probability_file
    return '/'.join( [ model, typ, str(nside), value, '%.4f' % z, which ] )

def KeySkymap( z, model, typ, nside, value ):
    ## get key in skymap_file
    return '/'.join( [ model, typ, str(nside), value, '%.4f' % z ] )

def KeyNearRay( model, nside, value ):
    return '/'.join( [ model, 'near',str(nside), key ] )


def FileNearRay( ipix ):
return root_rays + model + '/ray%i_%i.h5' % (ipix,npix)

            


### data correction


def RedshiftCorrectionFactor( redshifts, redshift_snap ):
    ## factor to correct redshift evolution
    ## data is given for redshift of snaphost, replace by redshift along LoS
    return ( 1 + redshifts ) / (1+redshift_snap)

def GetRedshiftCorrectionFactors( data, far=False, steps=None ):
    ## correct scaling factors to cure discontinuities due to change in snapshot
    ##   (values correct at steps, no evolution in snapshots)
    if steps is None:
        steps = [0]                                                       ## steps start with final redshift = 0
        if far:
            steps.extend( LocateSteps(data['Density']) )                  ## add locations of discontinuities
        steps.append( data['Density'].size-1 )                                          ## finally, add initial redshift
    elif not ( ( type(steps[0]) is int ) or ( type(steps[0]) is np.int ) ): 
        print "steps needs to be integer indices"

        
    a = []
    for s0, s1 in zip( steps, steps[1:] ):                            ## in each snapshot range,
        a.extend( RedshiftCorrectionFactor( data['redshift'][s0:s1+(s1==data['redshift'].size-1)], data['redshift'][s1-1 if np.round(np.abs(data['redshift'][s0]),3)>0 else 0] ) )
        ## choose norm at redshift of initial, correctly scaled value
#        a0 = 1./(1+data['redshift'][s1-1 if np.round(np.abs(data['redshift'][s0]),3)>0 else 0]) # snapshot at z=0 has correct value at lowest reshift, others at highest
#        a.extend( a0*( 1 + data['redshift'][s0:s1+(s1==data['redshift'].size-1)] ) ) ## compute the correcting scaling factors
    return np.array(a)





### FRB_Likelihood convenience
### !!! tidy until here

def ActualRedshift( dl, z0, z_snap, redshift_enter_last_snapshot, redshift_accuracy ):
    ## returns actual redshift of ray data when added to LoS with final redshift of z0
    ##   needs redshift of snapshot z_snap to correct the travelled length
    if z_snap < redshift_enter_last_snapshot:  ## when snapshot is later than time to enter final snapshot
        z_snap = 0.             ## use final redshift ( = 0 ) as reference (== redshift of snap)
                                ## !!! change this if your final redshift is not 0
    dt, dt_ = np.ones(dl.size), np.ones(dl.size)*1.5
#    z = 0.
    z = np.array([z0]*len(dl))  # first guess is intial redshift of segment
    z_ = z-1
    t0 = t_from_z( z0 ).value*1

    ## iteratively compute the correct redshift by
    ##   1. from travelled distance compute travelled time, scaled by current redshifts solution
    ##   2. compute redshift corresponding to travelled time
    ## finish when results are stable
    dt0 = np.cumsum( dl ) / speed_of_light.value * (1+z_snap) ## values used multiple times, comoving dl / c
    ## while not converged
    i = 0
#    while np.any( np.round( z_ / z, redshift_accuracy ) != 1) and i<3:
    while np.allclose( z_ , z, rtol=10.**-redshift_accuracy ) and i<3:
        i += 1
        z_ = z[:]
        ##   compute cumulative time travelled through cells
        ##   and the corresponding redshift along the full LoS
        z = np.array( map( lambda t: z_from_t( t ), t0 - dt0/(1+z) ) )  # LoS goes back in time, hence subtract from starting time the time travelled through cell to obtain correct redshift
    return z

import healpy as hp, time

t0 = time.time()

def GetChoppedRayData(model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_trans, i,  redshift_accuracy, correct=True ):
#def GetChoppedRayData(i, model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_trans, correct=True ):
    ## returns data of LoS combined from (randomly sampled) chopped rays
    ## generate random sampler
    if i % 30 == 0:
        print '%.4f  in %i seconds ' % ( float(i) / hp.nside2npix(nside), time.time() - t0 )
    RS = np.random.RandomState( i )
    steps = [0] # grab indices of snapshot changes for correction factor
    ## read data of first, constrained part of the LoS
    data_LoS = GetRayData( model, fields, nside, redshift_initial, i, far=False, correct=False )
    ## for each snapshot
    for z_snap, z_shot in zip( redshift_snapshots[1:], redshift_snapshots[2:] ):
        i_rays = range( N_choppers )
        ##   while LoS hasn't reached  full length
        while data_LoS['redshift'][-1] < z_shot:
            ##     get random ray index
            i_ray = i_rays[ RS.randint( len(i_rays) ) ]
            ##  !!!!   never use same index twice
            #                i_rays.remove( i_rays[r] )
            ##     read data of random ray
            data_ray = GetRayData(model, fields, nside, redshift_initial, i_ray, far=False, chopped=z_snap, correct=False )
            ##     correct the redshift
            data_ray['redshift'] = ActualRedshift( data_ray['dl'], data_LoS['redshift'][-1], z_snap, redshift_trans, redshift_accuracy )
            ##     add up on previous parts
            data_LoS = np.concatenate( [data_LoS, data_ray] )
        data_LoS = data_LoS[ data_LoS['redshift'] <= redshift_max ]
        steps.append( data_LoS['dl'].size )
        ## correct for z dependence
    if correct:
        a = GetRedshiftCorrectionFactors( data_LoS, steps=steps ) #np.array(redshift_snapshots)[np.arange(len(redshift_snapshots)) != 1] )
        for field in data_LoS.dtype.fields:
            data_LoS[field] *= ( a**-comoving_exponent[field]               ## data in proper cgs
                                 if not 'B' in field else
                                 a*( ( 1+data_LoS['redshift'][:] ) / (1+redshift_initial) )  ### B is written wrongly, hence needs special care
                             )
    return data_LoS


def GetChoppedSegmentData(model, fields, nside, redshift_initial, i_segment, redshift_snapshot, redshift_start, correct=True ):
#def GetChoppedRayData(i, model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_trans, correct=True ):
    ## returns data of LoS segment in z_snap snapshot, starting from z

    ## read raw data of output
    data = GetRayData(model, fields, nside, redshift_initial, i_segment, far=False, chopped=redshift_snapshot, correct=False )
    ## correct redshift
    data['redshift'] = ActualRedshift( data['dl'], redshift_start, redshift_snapshot, redshift_trans, redshift_accuracy )
    ## in case, correct data
    if correct:
        a = RedshiftCorrectionFactor( data['redshift'], redshift_snapshot )
        for field in data.dtype.fields:
            data[field] *= ( a**-comoving_exponent[field]               ## data in proper cgs
                                 if not 'B' in field else
                                 a*( ( 1+data_LoS['redshift'][:] ) / (1+redshift_initial) )  ### B is written wrongly, hence needs special care
                             )
    
    return data



def GetFarNearRayData(model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_trans, ipix, correct=True, directions=False, chopped=False ):
        ## reads and combines (scaling corrected) data of near (z=0) and far (z>0) LoS to single ordered arrays
    data_near = GetRayData(model, fields, nside, redshift_initial, ipix, model=model, far=False, correct=correct, directions=directions )
    if chopped:
        data_far = GetChoppedRayData(model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_trans, i, correct=correct )        
    else:
        data_far = GetRayData(model, fields, nside, redshift_initial, ipix, model=model, far=True, correct=correct, directions=directions )
    data = np.concatenate( (data_near, data_far) )
    ## sort by redshift
    data.sort( order='redshift')
    return data












def GetDMRM( data, redshifts ):
    ## return DMRM contributions between skymaps ( cumsum the result )
    DMRM = np.zeros( (2, len(redshifts)-1), 'float' )
    DM_RM = np.array( DispersionMeasure( data['Density'], data['dl'], data['redshift'], data['B_LoS'] ) )
    iz = 0
    for z0, z1 in zip( redshifts, redshifts[1:] ):  # z starts with 0
        ix_zs = (z0 <= data['redshift'])*(data['redshift'] < z1 )
        DMRM[:,iz] = np.sum( DM_RM[:,ix_zs], axis=1 )
        iz += 1
    return DMRM


#def GetRayDMRM( redshifts, Function_GetRayData, GetDataArgs, ipix ):
def GetRayDMRM( redshifts, Function_GetRayData, ipix ):
#    print Function_GetRayData( ipix )
#    return GetDMRM( GetChoppedRayData( ipix, *GetDataArgs ), redshifts )
    return GetDMRM( Function_GetRayData( ipix ), redshifts )






def MakeDMRMrayChopped( model, fields, nside, redshift_initial, redshift_skymaps, redshift_snapshots, ipix ):
    ## computes DM & RM values of chopped segments between redhsifts of skymaps and writes them to file
    
    DMRM = np.zeros( ( 2, len(redshift_skymaps)-1 ) )
    ## for each segment between redshifts
    for i in range( len(redshift_skymaps)-1 ):
    ##   read data of included part of LoS
        data = GetChoppedRayData_partly( model, fields, nside, redshift_initial, ipix, redshift_snapshots, redshift_skymaps[i], redshift_skymaps[i+1] )
    ##   compute total DM & RM of  part
        DMRM[:,i] = np.sum( DispersionMeasure( data['Density'], data['dl'], data['redshift'], data['B_LoS'] ), axis=1 )
    ## add up previous parts
    DMRM = np.cumsum( DMRM, axis=1 )
    ## collect in singla h5 file
    name = ['DM','RM']
    with h5.File( DMRMrays_file ) as f:
        for i in range(2):
            new_key = '/'.join(['',model,str(nside), 'chopped', str(ipix),name[i]] )
            try:
                f.__delitem__(new_key)
            except:
                pass
            f.create_dataset(new_key, data=DMRM[i] )
    return;
