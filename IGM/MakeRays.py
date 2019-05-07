import numpy as np, yt, os, h5py as h5, healpy as hp
from pathway import *
from Convenience import *
from Parameters import *
from Models import *

## path length within first snapshot in code units
path_length_code = ( comoving_radial_distance(dsz0.current_redshift, redshift_max_near)
                / dsz0.domain_width.in_cgs()[0]
).d 
        


def MakeNearRay( ipix, lr=None, start_position=observer_position, collect=True ):
    ## creates constrained part of LoS starting from observer in z=0 snapshot
    ## provide lr = Lightray( dsz0 ) for pool computation
    if lr is None:
        lr = LightRay( dsz0 )

    direction = np.array( hp.pix2vec( nside, ipix) ) + 0.0001 # shift to not move along edge of cells
    end_position = path_length_code * direction + start_position
    filename = root_rays + model + '/ray%i_%i.h5' % (ipix,npix),
    lr.make_light_ray(   ## compute LoS and save to .h5 file
        start_position = start_position,
        end_position = end_position, 
        data_filename = filename,
        fields = fields,
        use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
    )

    ### see if possible to write to same file in multi processing, else put in seperate function
    if collect:
        with h5.File( rays_file, 'a' ) as ff:
            for i in range(npix):
                with h5.File( filename ) as fil:
                    f = fil['grid']
                    for key in f.keys():
                        new_key = '/'.join(['',model,str(nside), 'far' if far else 'near', str(i),key] )
                        try:
                            ff.__delitem__(new_key)
                        except:
                            pass
                        ff.create_dataset(new_key, data=f[key].value )
                            
            os.remove( filename )
    





def MakeNearRays( start_position=observer_position ):  ## works
    ## computes LoS in the last (z=0) snapshot until they enter next snapshot 
    ##   in all directions from given start_position defined by a healpix map 
    ## saves them to .h5 files
    
    ## make LightRay object of first data set with redshift = 0
    lr = LightRay( dsz0 )
    for ipix in trange( npix ): ## in each direction in healpix map
        p = multiprocessing.Process( target=MakeNearRay, args=(ipix,) )
        p.start()
    p.join()




def MakeChoppedRay( i, i_snap, lr=None, collect=True  ):
    if lr is None:
        lr = LightRay( ts[-1-i_snap] )

    RS = np.random.RandomState( seed*( 1 + i + i_snap*N_choppers ) )

    ##     find start & end within border, such that distance > minimum trajectory length
    start_position, end_position = np.zeros([2,3])
    while np.linalg.norm( end_position - start_position ) < traj_length_min:  
        for j in range(3):
            start_position[j], end_position[j] = RS.random_sample(2) * ( border[1][j] - border[0][j] ) + border[0][j]

    ## compute LoS and save to .h5 file
    #                    lr = LightRay( ts[-1-i_snap] )  ## needed inside to have function pickable
    filename = root_rays + model + '/ray_z%1.2f_%i.h5' % ( redshift_snapshots[i_snap], i )
    lr.make_light_ray(   
        start_position = start_position,
        end_position = end_position, 
        data_filename = filename,
        fields = fields,
        use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
    )
    ## write to collecting file
    if collect:
        with h5.File( rays_file, 'a' ) as ff:         ## open target file
            with h5.File( filename, 'r' ) as fil:  ##     open ray file
                f = fil['grid']
                for key in f.keys():          ##     for each data field
                    new_key = '/'.join(['',model, '%1.2f' % z, 'chopped',  str(i),key] )
                    try:                      ##       delete if present
                        ff.__delitem__(new_key)
                    except:
                        pass                  ##       write ray data to target file
                    ff.create_dataset(new_key, data=f[key].value )
                        
        os.remove( filename )             ##       remove ray file
    return True

    

def MakeChoppedRays(): 
    ## produces N trajectories within each snapshot

    ## for each redshift of snapshots
    for i_snap, z in enumerate( redshift_snapshots[1:-1] ):
        ##   load the corresponding snapshot to LightRay
        lr = LightRay( ts[-1-i_snap] )
        for n in trange(N_choppers):
            p = multiprocessing.Process( target=MakeChoppedRay, args=(n,) )
            p.start()
        p.join()

def MakeFarRay( ipix, collect=True ):
    lr = LightRay(   # load LightRay model for each ray to use different seed
        param_file,  
        simulation_type="Enzo",
        near_redshift=redshift_max_near, ## in this redshift range  !!!!
        far_redshift=min( redshift_max, ds0.current_redshift ),
        find_outputs=True,     # find the snapshots
    )
    filename = root_rays + model + '/ray_far%i_%i.h5' % (i,npix) ,
    lr.make_light_ray( # compute LoS and save to .h5
        seed= ipix*seed,
        fields=fields,
        data_filename = filename,
        use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
    )
    ## write to collecting file
    if collect:
        with h5.File( rays_file, 'a' ) as ff:         ## open target file
            with h5.File( filename, 'r' ) as fil:  ##     open ray file
                f = fil['grid']
                for key in f.keys():          ##     for each data field
                    new_key = '/'.join(['',model, '%1.2f' % z, 'chopped',  str(i),key] )
                    try:                      ##       delete if present
                        ff.__delitem__(new_key)
                    except:
                        pass                  ##       write ray data to target file
                    ff.create_dataset(new_key, data=f[key].value )
                        
        os.remove( filename )             ##       remove ray file
    return;

def MakeFarRays(self) : 
    ## computes LoS in all z>0 snapshots 
    ##   random start_position and direction, one for each cell in a healpix map 
    ## saves them to .h5 files

    ## make LightRay object in all directions and snapshots of simulation
    for ipix in trange(npix):
        p = multiprocessing.Process( target=MakeFarRay, args=(ipix,) )
        p.start()
    p.join()

