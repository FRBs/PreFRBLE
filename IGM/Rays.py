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


def CollectRays( remove=True ):
    ## collect all files with single rays in constrained volume to rays_file
    for ipix in range( npix ):
        ## consider all possible directions
        CollectRay( ipix, remove=remove )



def CollectRay( ipix, remove=True ):
    ## write h5 of single ray to rays_file
    filename = FileNearRay( ipix, model=model, npix=npix )
    with h5.File( filename, 'r' ) as fil:  ##     open ray file
        f = fil['grid']
        for measure in f.keys():          ##     for each data field
            key = KeyNearRay( model_tag, nside, ipix, measure )
            Write2h5( rays_file, [f[measure].value], [key] )
    if remove:
        os.remove( filename )             ##       remove ray file




            
        

def MakeNearRay( ipix, lr=None, observer_position=observer_position, collect=False ):  
    ## reads data of LoS in constrained spherical volume around observer in z=0 snapshot
    ### provide lr = Lightray( dsz0 ) for pool computation
    
    ## write ray data to
    filename = FileNearRay( ipix, model=model, npix=npix )
    ## skip if ray was produced already
    if os.path.isfile( filename ):
        print 'skip %i' % ipix,
        return;

    if lr is None:
        lr = LightRay( dsz0 )  ## initialize trident.LightRay
    for off in [ 0., 0.0001, -0.0001]:  ## sometimes fails, when LoS is on axis. Try again with small offset
        try:
            direction = np.array( hp.pix2vec( nside, ipix) ) + off # shift to not move along edge of cells
            start_position = path_length_code * direction + observer_position
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



def MakeNearRays( observer_position=observer_position ):  ## works
    ## computes LoS in the last (z=0) snapshot until they leave the volume
    ##   in all directions from given start_position, direction defined by a healpix map, distance is maximum distance within constrained volume, i. e. 0.5*edgelength
    ## saves them to .h5 files
    
    ## make LightRay object of first data set with redshift = 0
    f = partial( MakeNearRay, observer_position=observer_position, collect=False )  ## define as pickleable function to allow multiprocessing


    ### !!!! this map ends too early, after producing the first round of rays... why ????
    pool = multiprocessing.Pool( N_workers_MakeNearRays )
    pool.map( f , range( npix ) )# , 1 )
#    map( f , range( npix ) )
    pool.close()
    pool.join()

    ## collect ray files to a single h5
#    CollectRays( remove=False )







def GetNearRayData( ipix, model=model, redshift_initial=redshift_initial, correct=True, B_LoS=True ):
    ## returns raw ray data of near ray in rays_file
    with h5.File( rays_file ) as f:
    ## open data of interest in rays file
        g = f[ '/'.join( [ keys['near'], str(ipix) ] ) ]
        z = g['redshift'].value

        ## use positive redshift ~ -z for z << 1   !!! remove, since not needed ???
        if np.round(z.min(),6) < 0:  
            z *= -1
#            g['redshift'][...] *= -1
            g['redshift'][...] -= g['redshift'][...].min()

        field_types = fields[:]                 ## requested fields
        field_types.extend(['redshift', 'dl'])  ## redshift and pathlengths within cell
        field_types.extend(['x', 'y', 'z'])     ## cell center positions
        if B_LoS:
            field_types.append('B_LoS')  ## line of sight magnetic field

        data = np.zeros(g['dl'].shape, dtype=[ (field, 'float') for field in field_types ])

        if correct:
            ## correct measures for smooth evolution of proper measures with redshift
            ##   correct scaling factor in final snapshot
            a = ScaleFactor( z, z_snaps[0] )

            ## for all fields of interest
            for field in field_types:
                ## except for B_LoS, which is not yet computed, hence not in g
                if not field is 'B_LoS':
                    ## apply the correct scaling factor
                    data[field] = g[field]* ( a**comoving_exponent[field]               ## data in proper cgs
                                              if not 'B' in field else
                                              ( ( 1+g['redshift'][:] ) / (1+redshift_initial) )/a  ### B is written wrongly, hence needs special care
                                          )

        else:
            ## read the fields of interest without redshift evolution
            for field in field_types:
                if not field is 'B_LoS':
                    data[field] = g[field]

            
        if B_LoS:  ## calculate magnetic field along LoS
            ## get the direction from healpix
            direction = np.array( hp.pix2vec( nside, ipix ) )
            data['B_LoS'] = GetBLoS( data, direction=direction )

        data.sort( order='redshift')
    return data





## Create LoS DM RM
def FilenameSegment( ipix, n ):
    return root_rays+model+'/ray_segment%03i_pix%05i.h5' % ( n, ipix )



def CreateSegment( lr, RS, redshift, n, ipix, length_minimum=0.5 ):
    ## create segment from LightRay object lr, with random orientation from random state RS, starting at redshift
    ## n is the number of the current segment, ipix is the index of the LoS it belongs to
    filename = FilenameSegment( ipix, n )
    ## find random start & end positin of the segment, with distance longer than length_minimum [relative to shortest border]
    length=0
    while length < length_minimum:
        ## find random start position within border
        start_position = RS.random_sample(3) * ( border[1] - border[0] ) + border[0]
        ## find random direction
        phi = RS.uniform( 0, 2*np.pi )
        theta = np.arccos( RS.uniform( -1, 1 ) )
        direction =  np.array( [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ] )
        ## find random length of the segment
        length = RS.uniform( min(border[1]-border[0])*length_minimum, np.linalg.norm( border[1]-border[0] ) )
        ## calculate end position
        end_position = start_position + length * direction
    
        ## check whether 
        ## reduce length, such that the LoS doesnt overshoot the probed volume
        for i in range(3): ## in each direction
            if border[0][i] > end_position[i]:   ## if border is exceeded
                length *=   ( border[0][i] - start_position[i] ) / ( length*direction[i] )  ## reduce length to hit that border, l = l * b_i/l_i
            elif border[1][i] < end_position[i]: ## at both sides
                length *= - ( border[1][i] - start_position[i] ) / ( length*direction[i] )  ## reduce length to hit that border, l = l * b_i/l_i
        ## correct end position with reduced length
        end_position = start_position + length * direction

        ## redo if length < length_minimum

#    print 'start', start_position, 'end', end_position, 'direction', direction, 'length', length
    ## create ray segment
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
def CreateChoppedLoSSegments( ipixs, redshift_snapshots=redshift_snapshots[:], redshift_max=redshift_max, ts=ts[:], redshift_accuracy=redshift_accuracy, seed=seed, force=False ):
    ## creates files with random segments of LoS data of ray through yt-TimeSeries ts that contains grid data of snapshots at redshift_snapshots
    ## ipixs is a list of the indices given to the rays, defines how many rays are produced


    ## if not forced to, produce ray only if its data isn't already written to file
    if not force:
        try:
            with h5.File( LoS_observables_file, 'r' ) as f:
                for ipix in ipixs[:]:
                    try:
                        tmp = f[ '/'.join( [ model, 'chopped',  str(ipix), 'DM'] ) ]
                        ipixs.remove(ipix)
                    except:
                        pass
        except:
            pass
        ## prevent the case of call eith empty ipixs
        if len(ipixs) == 0:
            return;

    ## exclude constrained near ray, go to z=0 instead  !!! remove completely
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
    ##    in later rounds: for each, read final redshift from previous and continue with that
                if not new:
                    redshift = h5.File( FilenameSegment( ipix, n[i_flag]-1 ) )['grid/redshift'].value[-1]
    ##    in case redshift is past next snapshot, skip and deflag
                    if redshift <= redshift_snapshots[i_snap]:  ###!!!!
                        flags[i_flag] = 0
                        continue
#                print 'make segment from z=%.4f' % redshift
                CreateSegment( lr, RS, redshift, n[i_flag], ipix )
                n[i_flag] += 1
    ##  round finished
            new = False

## to test the code, use:
#            break ### !!! create only one round of segments
#        break  ### !!!  only use first snapshot

    
    



## reduce rays to LoS observables at redshift of interest
def CreateLoSObservables( ipix, remove=True, redshift_snapshots=redshift_snapshots[:], plot=False, models=[model] ):
    ## collects segment files of ipix'th ray created by CreateChoppedLoSSegments
    ## computes and returns observables for all models, that are provided with a |B|~rho relation in relation_file
    ## results are computed for sources located at redshift_skymaps   !!! rename redshift_skymaps



    field_types = fields[:]     ## requested fields
    field_types.extend(['x', 'y', 'z','redshift', 'dl'])  ## add cell center positions, redshift and pathlengths within cells
    field_types.append( 'B_LoS' ) ## add magnetic field parallel to LoS

    ## exclude constrained near ray, go to z=0 instead  !!! remove completely
    try:
        redshift_snapshots.remove( redshift_max_near )
    except:
        pass
    
    ## create empty array for results
    results = np.zeros( (2+len(models),len(redshift_skymaps)-1) )

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
        return;
    
    for f in files:
        ## read file data
        try:
            g = h5.File(f)['grid']
        except:
            print f, "has no 'grid' "
            return;
        ## correct data (smooth redshift dependence and B, which is saved wrongly by ENZO)
        ##   find redshift of current snapshot.  snapshots are used for z < redshsift_snapshot, only the z=0 snapshot is used at higher redshift < redshift_trans. To account for that, compare to redshift_snapshots[1:], where redshift[1] is redshift_trans
        i_snap = np.where( np.round(redshift_snapshots[1:], redshift_accuracy) >= g['redshift'].value.max() )[0][0] ## !!! check if redshift_near is removed
        redshift_snapshot = z_snaps[ i_snap ]        
        ##   correction factor scales data from redshift_snapshot to redshift of cell
        a = ScaleFactor( g['redshift'].value, redshift_snapshot )
        ## prepare empty data array
        data = np.zeros(g['dl'].shape, dtype=[ (field, 'float') for field in field_types ])
        ## for all fields of interest
        for field in field_types:
            ## except B_LoS, which is not yet computed, hence not in g
            if not field is 'B_LoS':
                ## smooth scaling, use correct expansion rate
                data[field] = g[field].value* ( a**comoving_exponent[field]               ## data in proper units
                                            if not 'B' in field else
                                            ( ( 1+g['redshift'][:] ) / (1+redshift_initial) )/a  ### B is written wrongly, hence needs special care
                                        )
        if remove: ## remove file
            os.remove( f )

        ## use data only before enter next snapshot, this data is provided in the next segment
        data = data[ data['redshift'] > redshift_snapshots[i_snap] ]
        if len(data['redshift']) == 0:
            continue

        ## compute magnetic field parallel to LoS
        direction = GetDirection( g['x'].value, g['y'].value, g['z'].value ) ## correctly scaled data results in wrong direction, use raw data instead 
        data['B_LoS'] = GetBLoS( data, direction=direction )

        ## calculate observables for each cell   !!! add SM
        DM = DispersionMeasure( density=data['Density'], distance=data['dl'], redshift=data['redshift'] )
        RM = RotationMeasure( DM=DM, B_LoS=data['B_LoS'] )
        SM = ScatteringMeasure( density=data['Density'], distance=data['dl'], redshift=data['redshift'], outer_scale=outer_scale_0_IGM )

        ### compute RM for other models
        RMs = []
        for im, m in enumerate( models ):
            ## since RM propto B_LoS, apply B(rho) renormalization factor
            RMs.append( RM * renorm( im, data['Density'] / ( critical_density*omega_baryon*(1+data['redshift'])**3 ) ) ) ## density in terms of average (baryonic) density
            
        if plot:
            plt.loglog( 1+data['redshift'], DM )
            plt.plot( 1+data['redshift'], RM )
            plt.plot( 1+data['redshift'], SM )
            print data['redshift'][0], data['redshift'][-1]

        ## sum up to corresponding redshift of interest in redshift_skymaps
        for i_map in range( len( redshift_skymaps ) - 1 ):
            ## skip maps not covered by ray
            if redshift_skymaps[i_map] > redshift_snapshot or redshift_skymaps[i_map+1] < redshift_snapshots[i_snap] :
                continue
            ## find all  contributors in redshift range
            i_zs = np.where( (redshift_skymaps[i_map] <= data['redshift']) * (data['redshift'] < redshift_skymaps[i_map+1])  )[0]
            if len(i_zs) > 0:
                ## sum DM
                results[0,i_map] += np.sum( DM[i_zs] )
                results[1,i_map] += np.sum( SM[i_zs] )
                ## sum RM for different magnetic field models
                for im, m in enumerate( models ):
                    results[2+im,i_map] += np.sum( RMs[im][i_zs] )
        ## free memory
        data, DM, RM, SM, RMs = 0, 0, 0, 0, 0

    if plot:
        plt.show()

    return np.cumsum( results, axis=1 )  ## return cumulative result, as results at low redshift add up to result at high redshift


def CreateLoSsObservables( remove=True, redshift_snapshots=redshift_snapshots[:], models=[model], N_workers=32, bunch=128 ):
    ## collects all rays created by CreateChoppedLoSSegments, computes their observables and writes them to LoS_observables_file
    ## computes observables for all models, that are provided with a |B|~rho relation in relation_file
    ## N_workers processes work parallel on bunch segments 

    ## find all segment files and read their indices
    files = glob( root_rays+model+'/*segment000*.h5' )
    pixs = map( lambda f: int( f.split('pix')[-1].split('.h5')[0] ), files )
    pixs.sort()
    pixs = np.array(pixs)

    ## CreateLoSObservables does the actual job, use as pickleable function, feed it with the required keywords
    f = partial( CreateLoSObservables, remove=remove, models=models )

    ## loop through bunches of ray indices
    for i in range(0, len(pixs), bunch ):
        ipixs = np.arange( i, min([i+bunch,len(pixs)]) )
        ## compute their LoS observables in parallel
        pool = multiprocessing.Pool( N_workers )
        LoS_observables = pool.map( f , pixs[ipixs] )
#        LoS_observables = map( f , pixs[ipixs] )  ## to check when parallel fails
        pool.close()
        pool.join()
        ## and write the results to LoS_observables_file, for all rays and considered models
        for ipix, LoS in zip( pixs[ipixs], LoS_observables ):
            for im, m in enumerate( models ):
                CollectLoSObservables( LoS[np.array([0,1,1+im])], '/'.join( [ m, 'chopped',  str(ipix)] ), measures=['DM','SM','RM'] )



def CollectLoSObservables( observables, key, measures=['DM', 'SM', 'RM'] ):
    ## write observables to LoS_observables_file at key_new
    Write2h5( LoS_observables_file, observables, [ '/'.join( [ key, v ] ) for v in measures ] )

