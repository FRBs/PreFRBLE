import numpy as np, yt, os, h5py as h5, sys

from Physics import *
from Models import *



## Probability / Likelihood Functions

def ProbabilityFunction( v, bins=10  ):
    v_min = min(v)
    f = 0.3  ## factor to exceed non-zero range
    hist = list(histogram( v, bins=bins, range=( v_min*(1.-f)**np.sign(v_min) , max(v)*(1.+f) ), density=True ))
    hist[0] = hist[0].astype('f')/np.sum(hist[0])
    return tuple(hist)



def combine_Ps( P1, x1, P2, x2, loop=False ):
    ## combine p.d.f. with values P1 and P2 at (equally spaced) x1 and x2, resp.
    x = np.linspace(x1[0]+x2[0],x1[-1]+x2[-1], x1.size+x2.size-1)
    if loop:
        P = np.zeros(x.size)
        for i, Pi in zip( x1, P1 ):
            for j, Pj in zip( x2, P2 ):
                P[ i+j == x ] += Pi*Pj
    else:
        Ps = np.dot( P1.reshape(P1.size,1), P2.reshape(1,P2.size) )
        P=[]
        for i in range(P1.size+P2.size-1):
            P.append( np.sum( [ Ps[ min(i-j,P1.size-1), j+max(0,i-j-P1.size+1) ]  for j in range( max(0,1+i-P1.size), min(i+1,P2.size) ) ] ) )
        P = np.array( P )
    return P, x


def histogram( data, bins=10, range=None, density=None, log=False ):
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



## Physics

def DispersionMeasure( density, distance, redshift, B_LoS=None ):
    ### density in g/cm^3
    ### distance in kpc
    ###  B_LoS in G
    ## electron density = gas density / ( proton mass * electron weight )
#    DM = density / (proton_mass * 1.16) * distance / (1 + redshift ) / kpc2cm * 1e-3 # cgs to pc / cm**3
    DM = density / (proton_mass * 1.16) * distance / (1 + redshift ) / kpc2cm * 1e3 # cgs to pc / cm**3
    if B_LoS is None:
        return DM
#    RM = B_LoS * DM *1e3 *1e6 *1e4 # cgs to rad / m^2
    RM = 0.81 * B_LoS * DM / (1+redshift) * 1e6 # rad / m^2
    return DM, RM

def DispersionMeasureIntegral( density, distance, redshift, B_LoS=None, axis=None ):
    if B_LoS is None:
        DM = DispersionMeasure( density, distance, redshift )
        return np.sum( DM, axis=axis )
    DM, RM = DispersionMeasure( density, distance, redshift, B_LoS )
    return np.sum( DM, axis=axis ), np.sum( RM, axis=axis )

## Geometry 
def GetDirection( x, y, z ):
    d = np.array( [ x[-1] - x[0], y[-1] - y[0], z[-1] - z[0] ] )
    return d / np.linalg.norm(d)


def GetBLoS( data, direction=None ):
    ## compute the LoS magnetic field for data of single ray
    ## !!! needs to straighten out x,y,z for old far rays
    if direction is None:
        direction = GetDirection( data['x'], data['y'], data['z'] )
#    return direction[0]*data['Bx']
#    return data['By']
#    return direction[1]
#    return direction[2]*data['Bz']
    return np.dot( direction, np.array( [ data['Bx'], data['By'], data['Bz'] ] ) )


def GetDirections( data, span=1):
    # use average direction
    x, y, z= data['x'], data['y'], data['z']
    dx, dy, dz = np.diff(x[::3*span]), np.diff(y[::3*span]), np.diff(z[::3*span])
    dx = np.append(dx, dx[-3:])
    dy = np.append(dy, dy[-3:])
    dz = np.append(dz, dz[-3:])
    d = np.array( [ np.repeat(dx,3*span)[:x.size], np.repeat(dy,3*span)[:x.size], np.repeat(dz,3*span)[:x.size] ] )
    d /= np.sum( d, axis=0, keepdims=True )
    return d

## combination

def GetSamples( N, M, seed=42 ):
    ## choose N samples of size M from N*M elements
    RS = np.random.RandomState( seed )
    X = N*M
    indices = range( X )
    samples = []
    for n in range(N):
        sample = []
        for m in range(M):
            i = RS.randint( X )
            sample.append( indices[i] )
            indices.pop(i)
            X -= 1
        samples.append(sample)
    return samples


## yt convenience functions

def TimeSeries( model_dir ):
    ## Reads full data series of model in model_dir
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
    ## get redshifts of snapshots
    z_snaps = [ ds.current_redshift for ds in ts ]
    z_snaps.sort()
    try:
        i = np.where( np.round( z_snaps, redshift_accuracy ) >= redshift_max )[0][0]+1 
        redshift_snapshots = z_snaps[:i]
    except:
        redshift_snapshots = z_snaps[:]
#    redshift_snapshots = [ z for z in z_snaps if z <= redshift_max ]
    if np.round(redshift_snapshots[0],redshift_accuracy) == 0: # if final snapshot at z=0, use it from half time since previous snapshot
        redshift_snapshots.append( redshift_trans )
    else:
        redshift_snapshots.append( 0. )  ## use final snapshot until z=0
    redshift_snapshots.append( redshift_max_near )

#    if np.round(min(redshift_snapshots), 4) > 0:
#        redshift_snapshots.append( 0. )  ## final redshift is always 0

    redshift_snapshots.sort()
    return z_snaps, redshift_snapshots

def BoxFractions( ts, domain_width_code, redshift_snapshots ):
    ## get required max_box_fraction for each snapshot
    return [ int( np.ceil(
        (comoving_radial_distance(z0,z1)/min(ds.domain_width.in_units('cm'))).d / domain_width_code
    ) )
             for z0, z1, ds in zip( redshift_snapshots[1:], redshift_snapshots[2:], ts[::-1])
    ]



## file management

def Write2h5( filename, datas, keys ):
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
    return '/'.join( [ model, typ, str(nside), value, '%.4f' % z, which ] )

def KeySkymap( z, model, typ, nside, value ):
    return '/'.join( [ model, typ, str(nside), value, '%.4f' % z ] )


            


## data correction

def LocateSteps( v, N=1e3 ):
    ## locates discontinuities in 1D array :
    ##   these are where there is strong change in the data, i.e.  dv[i-1] << dv[i] >> dv[i+1]
    ##   locates all of these pronounced local maxima
    dv = np.abs(np.diff(v)) / v[:-1]
    return np.where( dv > 1. / N)[0] + 1 ## +1 to include step

def CorrectionFactors( redshifts, redshift_snap ):
    return ( 1 + redshifts ) / (1+redshift_snap)

def GetCorrectionFactors( data, far=False, steps=None ):
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
        a.extend( CorrectionFactors( data['redshift'][s0:s1+(s1==data['redshift'].size-1)], data['redshift'][s1-1 if np.round(np.abs(data['redshift'][s0]),3)>0 else 0] ) )
        ## choose norm at redshift of initial, correctly scaled value
#        a0 = 1./(1+data['redshift'][s1-1 if np.round(np.abs(data['redshift'][s0]),3)>0 else 0]) # snapshot at z=0 has correct value at lowest reshift, others at highest
#        a.extend( a0*( 1 + data['redshift'][s0:s1+(s1==data['redshift'].size-1)] ) ) ## compute the correcting scaling factors
    return np.array(a)





## FRB_Likelihood convenience

def GetRayData_old( model, fields, nside, redshift_initial, ipix, far=False, chopped=False, correct=True, directions=False, B_LoS=True ):
    # returns written data of ipix'th ray in proper cgs units
    with h5.File( rays_file ) as f:
        typ = 'near'
        if not ( chopped is False ):  ## chopped is redshift of snapshot from which to take ray
            typ = 'chopped'
        elif far:
            typ = 'far'
        print f['primordial'].keys()
        print '/'.join([model,str(nside) if chopped is False else '%1.2f' % chopped, typ, str(ipix)])
        g = f['/'.join([model,str(nside) if chopped is False else '%1.2f' % chopped, typ, str(ipix)])]
        if np.round(g['redshift'].min(),6) < 0:  ## use positive redshift ~ -z for z << 1
            g['redshift'][...] *= -1
        if correct:
            ## correcting scaling factor to obtain proper cgs units 
            ###  !! discontinuities due to change in snapshot
            a = GetCorrectionFactors( g, far=far )
        else:
            a = np.ones(g['redshift'].size)
        
        ## fields to read from the ray
        field_types = fields[:]                 ## requested fields
        field_types.extend(['redshift', 'dl'])  ## redshift and pathlengths
        field_types.extend(['x', 'y', 'z'])     ## cell center positions
        if directions:
            field_types.extend(['dx', 'dy', 'dz'])  ## path vector components ## wrongly written, all identical

        if B_LoS:
            field_types.append('B_LoS')  ## line of sight magnetic field
        

        # create ordered array
        # write all the fields
        # in case, add BLoS
        data = np.ones(g['dl'].shape, dtype=[ (field, 'float') for field in field_types ])
        for field in field_types:
            if not field is 'B_LoS':
                data[field] = g[field]* ( a**-comoving_exponent[field]               ## data in proper cgs
                                          if not 'B' in field else
                                          ( ( 1+g['redshift'][:] ) / (1+redshift_initial) )*a  ### B is written wrongly, hence needs special care
                                      ) ** correct
        if B_LoS:
            data['B_LoS'] = GetBLoS( data )

        data.sort( order='redshift')
    return data

#def ActualRedshift(redshift_last, data, z0, z_snap ):
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

def GetChoppedRayData(model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_last, i,  redshift_accuracy, correct=True ):
#def GetChoppedRayData(i, model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_last, correct=True ):
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
            data_ray['redshift'] = ActualRedshift( data_ray['dl'], data_LoS['redshift'][-1], z_snap, redshift_last, redshift_accuracy )
            ##     add up on previous parts
            data_LoS = np.concatenate( [data_LoS, data_ray] )
        data_LoS = data_LoS[ data_LoS['redshift'] <= redshift_max ]
        steps.append( data_LoS['dl'].size )
        ## correct for z dependence
    if correct:
        a = GetCorrectionFactors( data_LoS, steps=steps ) #np.array(redshift_snapshots)[np.arange(len(redshift_snapshots)) != 1] )
        for field in data_LoS.dtype.fields:
            data_LoS[field] *= ( a**-comoving_exponent[field]               ## data in proper cgs
                                 if not 'B' in field else
                                 a*( ( 1+data_LoS['redshift'][:] ) / (1+redshift_initial) )  ### B is written wrongly, hence needs special care
                             )
    return data_LoS


def GetChoppedSegmentData(model, fields, nside, redshift_initial, i_segment, redshift_snapshot, redshift_start, correct=True ):
#def GetChoppedRayData(i, model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_last, correct=True ):
    ## returns data of LoS segment in z_snap snapshot, starting from z

    ## read raw data of output
    data = GetRayData(model, fields, nside, redshift_initial, i_segment, far=False, chopped=redshift_snapshot, correct=False )
    ## correct redshift
    data['redshift'] = ActualRedshift( data['dl'], redshift_start, redshift_snapshot, redshift_last, redshift_accuracy )
    ## in case, correct data
    if correct:
        a = CorrectionFactors( data['redshift'], redshift_snapshot )
        for field in data.dtype.fields:
            data[field] *= ( a**-comoving_exponent[field]               ## data in proper cgs
                                 if not 'B' in field else
                                 a*( ( 1+data_LoS['redshift'][:] ) / (1+redshift_initial) )  ### B is written wrongly, hence needs special care
                             )
    
    return data



def GetFarNearRayData(model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_last, ipix, correct=True, directions=False, chopped=False ):
        ## reads and combines (scaling corrected) data of near (z=0) and far (z>0) LoS to single ordered arrays
    data_near = GetRayData(model, fields, nside, redshift_initial, ipix, model=model, far=False, correct=correct, directions=directions )
    if chopped:
        data_far = GetChoppedRayData(model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_last, i, correct=correct )        
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
