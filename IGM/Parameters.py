import healpy as hp, numpy as np, os
from Models import *
from Physics import *
from pathway import *
from Convenience import *

## specify parameters of the analysis
model_tag = 'test_border'
#model_tag = 'test_uniform'
model_tag = 'primordial'
nside = 2**6
redshift_max = 1.0
redshift_max = 6.0
#redshift_max = 10.0
#redshift_max = 99.0
observer_position = np.ones(3) * 0.5


N_skymaps_near = 10  ## number of skymaps taken form constrained ray


fields = ['Density','Bx', 'By', 'Bz']   ## fields of interest to calculate DM & RM

seed = 42

f_safety = 1.25  # factor to overestimate required number of segments in order to certainly reach path length
#f_safety = 5.  # factor to overestimate required number of segments in order to certainly reach path length
redshift_accuracy = 4 # parameter to round slightly off redshifts of snapshots

average_length_segment = 0.8 ## relative to size of considered volume

## model parameters
model = models[model_tag][0]
param_file = models[model_tag][1]
periodic = models[model_tag][2]
border = models[model_tag][3]
redshift_initial = models[model_tag][4]

## derived parameters
npix = hp.nside2npix(nside)
domain_width_code = border[1]-border[0] 
center = domain_width_code / 2 + border[0]
redshift_skymaps = np.arange(0,redshift_max+0.1,0.1) ## redshifts of skymaps for results
ts = TimeSeries( root_data + models[model_tag][0] )[::-1]
ds0 = ts[-1]  ## initial snapshot
dsz0 = ts[0]  ## final z=0 snapshot
dsz1 = ts[1]  ## pre-final snapshot
redshift_last = dsz1.current_redshift ## maximum redshift for LoS in (constrained) z=0 volume
cell_width_comoving = (dsz0.domain_width.in_units('cm')/dsz0.domain_dimensions)[0]
        # redshift half time between final two snapshots
redshift_trans = ds0.cosmology.z_from_t(  np.mean( [ ds0.cosmology.t_from_z(dsz0.current_redshift), ds0.cosmology.t_from_z(dsz1.current_redshift) ] ) )
        # redshift when ray enters the constrained volume ( sphere of 0.5 edgelenth of border from observer at center )
redshift_max_near = dsz0.cosmology.z_from_t( dsz0.cosmology.t_from_z( 0 ) - dsz0.domain_width.in_cgs()[0] * min( border[1] - border[0] ) / 2/ speed_of_light) # get z from time = final time minus travel time ( = distance / LightSpeed)
redshift_skymaps_near = np.linspace(0,redshift_max_near,1+N_skymaps_near)
magnitude_near_redshift = int(1 + np.ceil(-np.log10(redshift_skymaps_near[1])))
#redshift_initial = ds0.parameters['CosmologyInitialRedshift']  ## globally set initial redshift of simulation
#N_choppers = 4  ## set to npix * required segments * 1.1, req. segments for each snapshot
#N_choppers = 2*max( [ npix, max( BoxFraction() ) ] )
N_workers_MakeRays = 256  ## make reasonable choice, number of maximum performance
N_workers_ReadRays = 16  ##   "-"  , based on ray distance, resolution and memory

z_snaps, redshift_snapshots = RedshiftSnapshots( ts, redshift_max, redshift_max_near, redshift_trans, redshift_accuracy )

traj_length_min = min( border[1] - border[0] ) / 2                  ## = half box edgelength ( maximum length that doesn't exclude positions )
box_fractions = BoxFractions( ts, min(domain_width_code), redshift_snapshots )  ## number of volume transitions required for the path length in snapshot
N_choppers = np.ceil( np.array( box_fractions ) / average_length_segment * f_safety ).astype('int')  ## number of required segments, accounting for the average length of segments
N_choppers_total = N_choppers*npix
'''
print N_choppers
print 'get samples'
samples =  [ GetSamples( npix, N_choppers[i_snap], seed=seed ) for i_snap in range(len( redshift_snapshots[2:] ))]
print 'got samples'
'''
