'''

Procedures to
 - obtain data from cells along the LoS
 - compute DM, RM and SM LoS integrals
 - 

for 'near" (within constrained volume) and "chopped" (high redshift) LoS.
"chopped" LoS exceed the constrained volume, as they traverse the simulation volume multiple times. They are obtained by stacking randomly oriented segments. The LoS is chopped in pieces.

'''


import numpy as np, yt, os, h5py as h5, healpy as hp, multiprocessing, time
import matplotlib.pyplot as plt
from glob import glob
from functools import partial
from trident import LightRay
from tqdm import trange
from pathway import *
from Convenience import *
from Parameters import *
from Models import *

## path length within latest snapshot (z=0) in code units
#path_length_code = ( comoving_radial_distance(dsz0.current_redshift, redshift_max_near)
path_length_code = ( comoving_radial_distance(0, redshift_max_near)
                / dsz0.domain_width.in_cgs()[0]
).d 
        
if not os.path.isdir( root_rays+model ):
    os.mkdir( root_rays+model )

keys = {
    'near'    : '/'.join(['',model_tag, 'near',str(nside) ] ),
    'far'     : '/'.join(['',model_tag, 'far', str(nside) ] ),
    'chopped' : '/'.join(['',model_tag, 'chopped' ] ),
}

def CollectRay_old( filename, key_new, remove=True ):
    ## write h5 of single ray to rays_file
    with h5.File( filename, 'r' ) as fil:  ##     open ray file
        f = fil['grid']
        for key in f.keys():          ##     for each data field
            new_key = '/'.join( [ key_new, key ] )
            Write2h5( rays_file, [f[key].value], [new_key] )
    if remove:
        os.remove( filename )             ##       remove ray file

def CollectRays( typ, remove=True ):
    ## collect all files with single rays in constrained volume to rays_file
    if typ=='chopped':
        file_name = root_rays + model + '/ray_z'
        for i_snap, z in enumerate( redshift_snapshots[1:-1] ): 
            for i in range( N_choppers_total[i_snap] ):
                try:
                    CollectRay( file_name + '%1.2f_%i.h5' % ( z, i ),
                                '/'.join([keys['chopped'], '%1.2f' % z,  str(i) ] ),
                                remove=remove
                    )
                except:
                    print "%s doesn't exist" % ( file_name + '%1.2f_%i.h5' % ( z, i ) )
                                         
    else:
        file_name = root_rays + model + '/ray'
        for ipix in range( npix ):
            ## consider all possible directions
            CollectRay( ipix, remove=remove )
#            CollectRay( file_name+'%i_%i.h5' % (ipix,npix),
#                        '/'.join( [ keys[typ], str(ipix) ] )
#            )


def CollectRay( ipix, remove=True ):
    ## write h5 of single ray to rays_file
    filename = root_rays + model + '/ray%i_%i.h5' % (ipix,npix),
    with h5.File( filename, 'r' ) as fil:  ##     open ray file
        f = fil['grid']
        for key in f.keys():          ##     for each data field
            new_key = KeyNearRay( model_tag, nside, key )
            Write2h5( rays_file, [f[key].value], [new_key] )
    if remove:
        os.remove( filename )             ##       remove ray file




            
        

def MakeNearRay( ipix, lr=None, observer_position=observer_position, collect=False ):
    ## reads data of LoS in constrained spherical volume around observer in z=0 snapshot
    ### provide lr = Lightray( dsz0 ) for pool computation
    
    ## write ray data to
    filename = FileNearRay( ipix )
    ## skip if ray was produced already
    if os.path.isfile( filename ):
        print 'skip %i' % ipix,
        return;

    if lr is None:
        lr = LightRay( dsz0 )  ## initialize trident.LightRay
    for off in [ 0., 0.0001, -0.0001]:  ## sometimes fails, when LoS is on axis. Try again with small offset
        try:
            direction = np.array( hp.pix2vec( nside, ipix) ) + off # shift to not move along edge of cells
            start_position = path_length_code * direction + start_position
            lr.make_light_ray(   ## compute LoS and save to .h5 file
                start_position = start_position,  # LoS starts at the edge and ends at observer
                end_position = observer_position,  
                data_filename = filename,
                fields = fields[:],
                redshift = redshift_max_near,  ## starting redshift, chosen to end at observer with z=0
                use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
            )
            break
        except:
            continue

    if collect:
        ## write to collection file
        ### !!! not possible to write to same file in multi processing. Use separate function CollectRays
        CollectRay( ipix )


def MakeNearRays( start_position=observer_position ):  ## works
    ## computes LoS in the last (z=0) snapshot until they leave the volume
    ##   in all directions from given start_position defined by a healpix map 
    ## saves them to .h5 files
    
    ## make LightRay object of first data set with redshift = 0
    f = partial( MakeNearRay, start_position=start_position, collect=False )  ## define as pickleable function to allow multiprocessing

    pool = multiprocessing.Pool( N_workers_MakeNearRays )
    pool.map( f , trange( npix ) )# , 1 )
    pool.close()
    pool.join()

    ## collect ray files to a single h5
    CollectRays( 'near' )



def MakeChoppedRayData( i_snap, i, lr=None, filename=None  ):
#    if filename is None:
#        filename = root_rays + model + '/ray_z%1.2f_%i.h5' % ( redshift_snapshots[i_snap+1], i )
    if lr is None:
        lr = LightRay( ts[i_snap] )

    RS = np.random.RandomState( seed * ( 1 + i + i_snap*N_choppers[i_snap] ) )

    ##     find start & end within border, such that distance > minimum trajectory length
    start_position, end_position = np.zeros((2,3))
    while np.linalg.norm( end_position - start_position ) < traj_length_min:  
        for j in range(3):
            start_position[j], end_position[j] = RS.random_sample(2) * ( border[1][j] - border[0][j] ) + border[0][j]

    ## compute LoS and save to .h5 file
    return lr.make_light_ray(   
        start_position = start_position,
        end_position = end_position, 
        data_filename = filename,
        fields = fields[:],
        use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
    )

def MakeChoppedRay( i_snap, i, lr=None, collect=False, force=False  ):
    filename = root_rays + model + '/ray_z%1.2f_%i.h5' % ( redshift_snapshots[i_snap+1], i )
    key = '/'.join( [ keys['chopped'], '%1.2f' % redshift_snapshots[i_snap+1],  str(i) ] )
    ## check if file exists
    exists = os.path.isfile( filename )
    with h5.File( rays_file ) as f:
        try:  ## see if is stored in collected file
            g = f[key]
            exists = True
        except:
            pass

    ## compute if it hasn't beend computed before or you force it
    if exists and not force:
        print 'I already know %s' % key
    else:
        MakeChopperRayData( i_snap, i, lr=lr, filename=filename )
        if lr is None:
            lr = LightRay( ts[i_snap] )

        RS = np.random.RandomState( seed * ( 1 + i + i_snap*N_choppers[i_snap] ) )

        ##     find start & end within border, such that distance > minimum trajectory length
        start_position, end_position = np.zeros((2,3))
        while np.linalg.norm( end_position - start_position ) < traj_length_min:  
            for j in range(3):
                start_position[j], end_position[j] = RS.random_sample(2) * ( border[1][j] - border[0][j] ) + border[0][j]

        ## compute LoS and save to .h5 file
        lr.make_light_ray(   
            start_position = start_position,
            end_position = end_position, 
            data_filename = filename,
            fields = fields[:],
            use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
        )
        ## write to collecting file
        if collect:
            CollectRay( filename, key )




def MakeChoppedRaysSnapshot( i_snap, pool=None, force=False ): 
    ## produces N trajectories within given snapshot
    ##   load the corresponding snapshot to LightRay
    no_pool = False
    if pool is None:
        pool = multiprocessing.Pool()
        no_pool = True

    f = partial( MakeChoppedRay, i_snap, force=force ) #, lr=lr )
    pool.map( f , trange( N_choppers_total[i_snap] ) )

    if no_pool:
        pool.close()
        pool.join()
    return pool

def MakeChoppedRays( force=False ): 
    ## produces N trajectories within each snapshot

    pool = multiprocessing.Pool()
    pools = []
    for i_snap in trange( len(redshift_snapshots[1:-1]) ):
        pools.append( MakeChoppedRaysSnapshot( i_snap, pool=pool, force=force ) )
    for p in pools:
        p.close()
        p.join()
    time.sleep(5)
    CollectRays( 'chopped' )

        
def MakeFarRay( ipix, collect=True ):
    lr = LightRay(   # load LightRay model for each ray to use different seed
        param_file,  
        simulation_type="Enzo",
        near_redshift=redshift_max_near, ## in this redshift range  !!!!
        far_redshift=min( redshift_max, ds0.current_redshift ),
        find_outputs=True,     # find the snapshots
    )
    filename = root_rays + model + '/ray_far%i_%i.h5' % (i,npix)
    lr.make_light_ray( # compute LoS and save to .h5
        seed= ipix*seed,
        fields=fields[:],
        data_filename = filename,
        use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
    )
    ## write to collecting file
    if collect:
        CollectRay( filename, '/'.join( [ keys['far'],  str(ipix) ] ) )
    return;

def MakeFarRays(self) : 
    ## computes LoS in all z>0 snapshots 
    ##   random start_position and direction, one for each cell in a healpix map 
    ## saves them to .h5 files

    ## make LightRay object in all directions and snapshots of simulation
    f = partial( MakeFarRay, collect=False )
    pool = multiprocessing.Pool()
    pool.map( f, trange( npix ) )
    pool.close()
    pool.join()
    CollectRays( 'far' )



def GetRayData( ipix, typ, model=model, redshift_initial=redshift_initial, correct=True, B_LoS=True ):
    ## returns full ray data of far or near ray or chopped segment ( typ indicates redshift of snapshot
    with h5.File( rays_file ) as f:
    ## open data of interest in rays file
        if typ in ['far','near']:
            g = f[ '/'.join( [ keys[typ], str(ipix) ] ) ]
        else:
            g = f[ '/'.join( [ keys['chopped'], '%1.2f' % typ, str(ipix) ] ) ]
            correct=False
        z = g['redshift'].value

        ## use positive redshift ~ -z for z << 1
        if np.round(z.min(),6) < 0:  
            z *= -1
#            g['redshift'][...] *= -1
            g['redshift'][...] -= g['redshift'][...].min()

        field_types = fields[:]                 ## requested fields
        field_types.extend(['redshift', 'dl'])  ## redshift and pathlengths
        field_types.extend(['x', 'y', 'z'])     ## cell center positions
        if B_LoS:
            field_types.append('B_LoS')  ## line of sight magnetic field

        data = np.zeros(g['dl'].shape, dtype=[ (field, 'float') for field in field_types ])

        if correct:
            ## correcting scaling factor to obtain proper cgs units 
            ###  !! discontinuities due to change in snapshot
            a = GetRedshiftCorrectionFactors( g, far=typ == 'far' )

            for field in field_types:
                if not field is 'B_LoS':
                    data[field] = g[field]* ( a**-comoving_exponent[field]               ## data in proper cgs
                                              if not 'B' in field else
                                              ( ( 1+g['redshift'][:] ) / (1+redshift_initial) )*a  ### B is written wrongly, hence needs special care
                                          ) #** correct

        else:
            for field in field_types:
                if not field is 'B_LoS':
                    data[field] = g[field]

            
        if B_LoS:
            if typ == 'near':  ## for constrained ray, get the direction from healpix
                direction = np.array( hp.pix2vec( nside, ipix ) )
            else:
                direction = GetDirection( g['x'].value, g['y'].value, g['z'].value )  ## correctly scaled data results in wrong direction, use raw data instead
            data['B_LoS'] = GetBLoS( data, direction=direction )

        data.sort( order='redshift')
    return data


## !!!!! think different: saving rays takes too much space, compute DMRMray on the fly and write that to file

def GetChoppedRayData( ipix, i_snap, model=model, redshift_initial=redshift_initial, B_LoS=True ):

    filename = root_rays + model + '/ray_z%1.2f_%i.h5' % ( redshift_snapshots[i_snap+1], ipix )
    data = MakeChoppedRayData( i_snap, ipix )
    if B_LoS:
        data['B_LoS'] = GetBLoS( data )

    data.sort( order='redshift')
    return data


            
def GetChoppedSegmentData( i_segment, i_snap, redshift_start, model=model, redshift_initial=redshift_initial, correct=True ):
#def GetChoppedRayData(i, model, fields, nside, redshift_initial, redshift_snapshots, N_choppers, redshift_max, redshift_trans, correct=True ):
    ## returns data of LoS segment in z_snap snapshot, starting from z

    redshift_snapshot = z_snaps[ i_snap ]
    ## read raw data of output
    data = GetChoppedRayData(i_segment, i_snap, model=model, redshift_initial=redshift_initial )
    ## correct redshift
    data['redshift'] = ActualRedshift( data['dl'], redshift_start, redshift_snapshot, redshift_trans, redshift_accuracy )
    ## in case, correct data
    if correct:
        a = RedshiftCorrectionFactor( data['redshift'], redshift_snapshot )
        for field in data.dtype.fields:
            data[field] *= ( a**-comoving_exponent[field]               ## data in proper cgs
                                 if not 'B' in field else ### B is written wrongly, hence needs special care
                                 a*( ( 1+data['redshift'][:] ) / (1+redshift_initial) ) 
                             )
    
    return data


def GetChoppedRayData_partly( ipix, redshift_start, redshift_end, sample, model=model, redshift_initial=redshift_initial, redshift_snapshots=redshift_snapshots, correct=True ):
    ## returns data of LoS from redshift_start to _end composed of randomly oriented segments

    ## if redshift_start = 0: start with near (constrained) ray
    z0 = redshift_start
    if z0 == 0:
        data = GetRayData( ipix, 'near', model=model, redshift_initial=redshift_initial )
        z0 = data['redshift'][-1]
    else:
        data = None
    ## while LoS hasn't reached full length
    while z0 < redshift_end:
    ##   read data of following ray
    ##     needs redshift when snapshot was written, this is first redshift of snapshots that is greater than the segments. Except for first snapshot, which is used until redshift_snapshots[1] and has z=0
        i_snap = np.where( np.array( redshift_snapshots[2:] ) > z0 )[0][0]
        z_snap = z_snaps[ i_snap ]
#        z_snap = redshift_snapshots[ ix ] if ix > 2 else 0.
        data_segment = GetChoppedSegmentData( ipix, i_snap, z0, model=model, redshift_initial=redshift_initial )
    ##   concatenate with previous data
        data = np.concatenate( [ data, data_segment ] ) if data is not None else data_segment
        ## if segment overshoots redshift of snapshot, cut that (instead continue with following snapshot
        if data_segment['redshift'][-1] > z_snap:
            data = data[ data['redshift'] <= z_snap ]
        
        z0 = data['redshift'][-1]
        print 'current redshift %f' % z0
    ## return data of LoS ( cut off part of final segment that overshoots the distance )
    return data[ data['redshift'] <= redshift_end ]


def CollectDMRMRay( DMRM, key_new ):
    Write2h5( DMRMrays_file, DMRM, [ '/'.join( [ key_new, key ] ) for key in ['DM','RM'] ] )

def MakeDMRMRay( ipix, collect=True ):
#    global samples
#    sample = samples[ipix][:]
    ## returns DM & RM along a single LoS ( chopped in segments )
    DMRM = np.zeros( [2, len(redshift_skymaps[1:])] )
    iz = 0
    for z0, z in zip( redshift_skymaps, redshift_skymaps[1:] ):
        data = GetChoppedRayData_partly( ipix, z0, z1, sample )
        DMRM[:,iz] = GetDMRM( data, [z0,z1] )
        iz += 1
    if collect:
        CollectDMRMRay( DM, '/'.join( [ keys['chopped'],  str(ipix)] ) )
    return DMRM

def MakeDMRMRays():
    pool = multiprocessing.Pool()
    pool.map(  MakeDMRMRay, trange( npix ) )












from trident import make_simple_ray

def GetDataChoppedSegment( i, ipix, i_snap, redshift, correct=True, B_LoS=True, lr=None, RS=None ):
    ## return data array of i'th ray segment in i_snap'th snapshot starting at redshift (going forward in time)
    if RS is None:
        RS = np.random.RandomState( seed * ( 1 + i + i_snap*N_choppers[i_snap] ) )
    start_position, end_position = np.zeros((2,3))
    while np.linalg.norm( end_position - start_position ) < traj_length_min:
            for j in range(3):
                start_position[j], end_position[j] = RS.random_sample(2) * ( border[1][j] - border[0][j] ) + border[0][j]
    field_types = fields[:]                 ## requested fields
    field_types.extend(['redshift', 'dl'])  ## redshift and pathlengths
    field_types.extend(['x', 'y', 'z'])     ## cell center positions
    if lr is None:
        ray = make_simple_ray(
            ts[i_snap],
            start_position=start_position,
            end_position=end_position,
            fields=[ ('enzo',field) for field in fields[:] ],
            redshift=redshift
        )
        ## ray -> data (ordered array of fields of interest)
        g = ray.covering_grid( 0, [0.,0.,0.], ray.domain_dimensions, field_types)
    else:
        filename = "%s%s/ray_seg%i.h5" % ( root_rays, model, RS.randint( 9999999999 ) )
#        filename = "%s%s/ray_seg%i.h5" % ( root_rays, model, 1 + i + i_snap*N_choppers[i_snap*(1+ipix)  ] ) 
        lr.make_light_ray(
            start_position=start_position,
            end_position=end_position,
            fields=[ ('enzo',field) for field in fields[:] ],
            redshift=redshift,
            data_filename=filename
        )
        g = h5.File( filename )['grid']
        os.remove( filename )

    if B_LoS:
        field_types.append('B_LoS')  ## line of sight magnetic field                

    data = np.zeros(g['dl'].shape, dtype=[ (field, 'float') for field in field_types ])
    ## correct data (redshift dependence and magnetic field error)
    if correct:
        ## correcting scaling factor to obtain proper cgs units
        a = GetRedshiftCorrectionFactors( g )

        for field in field_types:
            if not field is 'B_LoS':
                data[field] = g[field].value* ( a**-comoving_exponent[field]               ## data in proper cgs                                                                                               
                                            if not 'B' in field else
                                            ( ( 1+g['redshift'][:] ) / (1+redshift_initial) )*a  ### B is written wrongly, hence needs special care                                                        
                                        ) ** correct

        else:
            for field in field_types:
                if not field is 'B_LoS':
                    data[field] = g[field].value

    ## add B_LoS
    if B_LoS:
        data['B_LoS'] = GetBLoS( data )
#        ray.add_field( 'B_LoS', function=AddBLoS )  ## !!!!

    return data




def MakeChoppedDMRMRay(  ipix, force=False ):
    ## reads full chopped LoS from simulation data, computes the observables of the segments between redshift of interest and writes them to DMRMrays_file
    print 'MakeRay'
    ## check whether ray was computed already
    if not force:
        key = '/'.join( ['',model,'chopped',str(seed),str(ipix),'DM'] )
        try:
            d = h5.File( DMRMrays_file )[key]
            print 'already got %s' % key
            return;
        except:
            pass
    ## produces DMRM values between redshifts of skymaps
    DMRM = np.zeros([2,len(redshift_skymaps)-1])
    ## starting from highest redshift
    i_snap_ = -1
    RS = np.random.RandomState( seed*( 1+ipix) )
    for i_map in range(len(redshift_skymaps)-1)[::-1]:
        z = redshift_skymaps[i_map+1]
    ## find index of corresponding snapshot ( two additional redshifts added between last two  snapshots, ignore first two to find correct index )
        i_snap = np.where( np.round( redshift_snapshots[2:], redshift_accuracy ) >= z )[0][0]
        if i_snap != i_snap_: # when enter new snapshot, initiate new LightRay object
            lr = LightRay( ts[i_snap] )
            i_snap_ = i_snap
            i_segment = 0
        i_segment += 1
#            sample = samples[i_snap][ipix][:]
    ## until redshift of next skymap is reached
        while z > redshift_skymaps[i_map]:
    ## get data of chopped segments
            data = GetDataChoppedSegment( i_segment, ipix, i_snap, z, lr=lr, RS=RS )
#            data = GetDataChoppedSegment( RS.randint( npix ), i_snap, z, lr=lr, RS=RS )
            z = data['redshift'][-1]
    ## cut possible overshoot
            if z < redshift_skymaps[i_map]:
                data = data[ data['redshift'] > redshift_skymaps[i_map] ]
                z = redshift_skymaps[i_map]
    ## add up DM and RM
        DMRM[:,i_map] +=  DispersionMeasureIntegral( data['Density'], data['dl'], data['redshift'], data['B_LoS'] )
#        DMRM[:,i_map] +=  DispersionMeasure( data['Density'], data['dl'], data['redshift'], data['B_LoS'] )
        print 'redshift', z, 'snap', i_snap, 'redshift snap', redshift_snapshots[i_snap+2]
    ## save to file
    ## collect files afterwards to allow parallel computation
    filename = [ '%s%s/ray%s_%i.dat' % ( root_rays, model, typ, ipix ) for typ in ['DM', 'RM'] ]
    for v, f in zip( DMRM, filename ):
        v.tofile( f ) #, sep=' ' )
    return;

def CollectChoppedDMRMRays():
    ## collect all files with single rays at high redshift to DMRMrays_file
    for typ in ['DM','RM']:
        for f_ray in glob( root_rays+model+'/*%s*.dat' % typ ):
            ipix = f_ray.split('_')[-1].split('.')[0]
            key = '/'.join( ['',model,'chopped',str(seed),str(ipix),typ] )
            Write2h5( DMRMrays_file, [np.fromfile( f_ray )], [key] )
            os.remove( f_ray )


def MakeChoppedDMRMRays( part=None ):
    ## create segments of LoS through several snapshots at different redshift
    pool = multiprocessing.Pool(N_workers_MakeChoppedRays)
    if part is None:
        ## create as many LoS as cells on the healpix skymap !!! not recommended, requires huge amount of storage 
        r = range( npix )
    else:
        ## optionally create LoS numbered by part = [ipix_start,ipix_end+1]
        r = range( *part )
    pool.map( MakeChoppedDMRMRay, r )
    pool.close()
    pool.join()
    










## Create LoS DM RM
def FilenameSegment( ipix, n ):
    return root_rays+model+'/ray_segment%03i_pix%05i.h5' % ( n, ipix )


def CreateSegment( lr, RS, redshift, n, ipix ):
    filename = FilenameSegment( ipix, n )
    length=0
    while length < 0.5:
        start_position = RS.random_sample(3) * ( border[1] - border[0] ) + border[0]
        phi = RS.uniform( 0, 2*np.pi )
        theta = np.arccos( RS.uniform( -1, 1 ) )
#        theta = RS.uniform( 0, np.pi )
#        phi = RS.uniform( 1e-6, 2*np.pi-1e-6 )
#        theta = np.arccos( RS.uniform( -1+1e-6, 1-1e-6 ) )
        length = RS.uniform( min(border[1]-border[0])/2, np.linalg.norm( border[1]-border[0] ) )
        direction =  np.array( [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ] )
        end_position = start_position + length * direction
    
        ## reduce length, such that the LoS doesnt overshoot the probed volume
        for i in range(3): ## in each direction
            if border[0][i] > end_position[i]:   ## if border is exceeded
                length = min( [ length, ( border[0][i] - start_position[i] ) / direction[i] ] )  ## reduce length to hit that border
            elif border[1][i] < end_position[i]: ## at both sides
                length = min( [ length, ( border[1][i] - start_position[i] ) / direction[i] ] )
        ## correct end position with reduced length
        end_position = start_position + length * direction
#        print 'start', start_position, 'end', end_position, 'direction', direction, 'length', length
        
    lr.make_light_ray(
        start_position=start_position,
        end_position=end_position,
        fields=[ ('enzo',field) for field in fields[:] ],
        redshift=redshift,
        data_filename=filename,
        use_peculiar_velocity=False,
#        njobs=32,
    )
    return;

## create rays
#def CreateChoppedRaySegments( ipixs ):
def CreateChoppedRaySegments( ipixs, redshift_snapshots=redshift_snapshots[:], redshift_max=redshift_max, ts=ts[:], redshift_accuracy=redshift_accuracy, seed=seed, force=False ):

    ## if not forced to, produce ray only if its data isn't already written to file
    if not force:
        try:
            with h5.File( DMRMrays_file, 'r' ) as f:
                for ipix in ipixs[:]:
                    try:
                        tmp = f[ '/'.join( [ model, 'chopped',  str(ipix), 'DM'] ) ]
                        ipixs.remove(ipix)
                    except:
                        pass
        except:
            pass
        if len(ipixs) == 0:
            return;

    ## exclude constrained near ray, go to z=0 instead
    try:
        redshift_snapshots.remove( redshift_max_near )
    except:
        pass

    RS = np.random.RandomState( seed * ( 1 + ipixs[0] ) )    
    ## index of earliest snapshot to be probed
    n_snaps = np.where( np.round( redshift_snapshots, redshift_accuracy ) >= redshift_max )[0][0]
    redshift = redshift_max
    n = np.zeros( len(ipixs) )
    ## for all snapshots, starting with the earliest
    for i_snap in range(n_snaps)[::-1]:
    ## load snapshot
        lr = LightRay( ts[i_snap] )
        redshift = redshift_snapshots[1:][i_snap]
        new = True
        flags = np.ones( len(ipixs) ) ## flag whether more segments are needed
        while np.any( flags ):
    ##  create round of segments
            for i_flag, ipix in enumerate(ipixs):
    ##    skip those that finished current snapshot
                if not flags[i_flag]:
                    continue
    ##    in later rounds, for each read final redshift from previous and continue with that
                if not new:
                    redshift = h5.File( FilenameSegment( ipix, n[i_flag]-1 ) )['grid/redshift'].value[-1]
    ##    in case redshift is past next snapshot, skip and deflag
                    if redshift <= redshift_snapshots[i_snap]:  ###!!!!
                        flags[i_flag] = 0
                        continue
                print 'make segment from z=%.4f' % redshift
                CreateSegment( lr, RS, redshift, n[i_flag], ipix )
                n[i_flag] += 1
    ##  round finished
            new = False

#            break ### !!! create only one round of segments
#        break  ### !!!  only use first snapshot

    
    


## reduce rays to LoS observables at redshift of interest
def CreateLoSDMRM( ipix, remove=True, redshift_snapshots=redshift_snapshots[:], plot=False, models=[model] ):
    field_types = fields[:]     ## requested fields
    field_types.extend(['x', 'y', 'z','redshift', 'dl'])  ## cell center positions, redshift and pathlengths
    field_types.append( 'B_LoS' )

    ## exclude constrained near ray, go to z=0 instead
    try:
        redshift_snapshots.remove( redshift_max_near )
    except:
        pass
    
    DMRM_ray = np.zeros( (1+len(models),len(redshift_skymaps)-1) )

    ## define B-rho-relation function for all models, given in relation_file
    f_renorm = [ np.genfromtxt( relation_file % m, names=True ) for m in models ]
    def renorm( i, rho ):
        f_rho = f_renorm[i]['density']
        ix = np.array( [ np.where( f_renorm[i]['density'] >= rhoi )[0][0] for rhoi in rho ] )
        return f_renorm[i]['Renorm'][ix]


    ## find all segment files of the ray
    files = glob( FilenameSegment( ipix, -12345).replace('-12345','*') )
    files.sort()

    ## check whether the final segment reaches z=0, else don't compute
    z0 = h5.File(files[-1])['grid/redshift'].value.min()
    if z0 > 0:
        print ipix, 'is not complete'
        return
    
    for f in files:
        ## read file data
        try:
            g = h5.File(f)['grid']
        except:
            print f, "has no 'grid' "
            continue
        ## correct data (redshift dependence and B)
        i_snap = np.where( np.round(redshift_snapshots[1:], redshift_accuracy) >= g['redshift'].value.max() )[0][0]
        redshift_snapshot = z_snaps[ i_snap ]        
        a = RedshiftCorrectionFactor( g['redshift'].value, redshift_snapshot )

        data = np.zeros(g['dl'].shape, dtype=[ (field, 'float') for field in field_types ])
        for field in field_types:
            if not field is 'B_LoS':
                data[field] = g[field].value* ( a**-comoving_exponent[field]               ## data in proper cgs
                                            if not 'B' in field else
                                            ( ( 1+g['redshift'][:] ) / (1+redshift_initial) )*a  ### B is written wrongly, hence needs special care
                                        )
        if remove:
            os.remove( f )
        ## use data only before enter next snapshot
        data = data[ data['redshift'] > redshift_snapshots[i_snap] ]
        if len(data['redshift']) == 0:
            continue

        ## correctly scaled data results in wrong direction, use raw data instead 
        direction = GetDirection( g['x'].value, g['y'].value, g['z'].value )
        data['B_LoS'] = GetBLoS( data, direction=direction )
        ## calculate DM RM
        DMRM = DispersionMeasure( data['Density'], data['dl'], data['redshift'], data['B_LoS'] )

        ### compute RM for other models
        RMs = []
        for im, m in enumerate( models ):
    ##      apply renormalization factor
            RMs.append( DMRM[1] * renorm( im, data['Density'] / ( critical_density*OmegaBaryon*(1+data['redshift'])**3 ) ) ) ## density in terms of average (baryonic) density
            
        if plot:
            plt.loglog( 1+data['redshift'], DMRM[0] )
            plt.plot( 1+data['redshift'], DMRM[1] )
            print data['redshift'][0], data['redshift'][-1]

        ## sum up to skymaps
        for i_map in range( len( redshift_skymaps ) - 1 ):
        ## skip maps not covered by ray
            if redshift_skymaps[i_map] > redshift_snapshot or redshift_skymaps[i_map+1] < redshift_snapshots[i_snap] :
                continue
        ##    find all  contributors in redshift range
            i_zs = np.where( (redshift_skymaps[i_map] <= data['redshift']) * (data['redshift'] < redshift_skymaps[i_map+1])  )[0]
            #### modify such that all models are computed
            if len(i_zs) > 0:
#                DMRM_ray[:,i_map] += np.sum( np.array(DMRM)[:,i_zs], axis=1 )
                DMRM_ray[0,i_map] += np.sum( DMRM[0][i_zs] )
                for im, m in enumerate( models ):
                    DMRM_ray[1+im,i_map] += np.sum( RMs[im][i_zs] )
        ## free memory
        data, RMs = 0, 0

    if plot:
        plt.show()

    return DMRM_ray

#    CollectLoSDMRM( DMRM_ray, '/'.join( [ model, 'chopped',  str(ipix)] ) )
#    for im, m in enumerate( models ):
#        CollectLoSDMRM( DMRM_ray[np.array([0,1+im])], '/'.join( [ m, 'chopped',  str(ipix)] ) )
    

def CreateLoSsDMRM( remove=True, redshift_snapshots=redshift_snapshots[:], models=[model], N_workers=32, bunch=128 ):
    ## find all segmented LoSs
    files = glob( root_rays+model+'/*segment000*.h5' )
    pixs = map( lambda f: int( f.split('pix')[-1].split('.h5')[0] ), files )
    pixs.sort()
    pixs = np.array(pixs)

    f = partial( CreateLoSDMRM, remove=remove, models=models )

    for i in range(0, len(pixs), bunch ):
        pool = multiprocessing.Pool( N_workers )
        ipixs = np.arange( i, min([i+bunch,len(pixs)]) )
        DMRM_rays = pool.map( f , pixs[ipixs] )
#        DMRM_rays = map( f , pixs[ipixs] )
        pool.close()
        pool.join()
        for ipix, ray in zip( pixs[ipixs], DMRM_rays ):
            for im, m in enumerate( models ):
                CollectLoSDMRM( ray[np.array([0,1+im])], '/'.join( [ m, 'chopped',  str(ipix)] ) )


def CollectLoSDMRM( DMRM, key_new ):
    ## write observables DMRM to DMRMrays_file at key_new
    Write2h5( DMRMrays_file, DMRM, [ '/'.join( [ key_new, key ] ) for key in ['DM','RM'] ] )

