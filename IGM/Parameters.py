'''
List of Parameters used for the analysis

'''


import healpy as hp, numpy as np, os
from Models import *
from Physics import *
from pathway import *
from Convenience import *

## specify parameters of the analysis
model_tag = 'primordial'
nside = 2**6                         ## healpix resolution parameter
redshift_max = 6.0                   ## maximum redshift of LoS
observer_position = np.ones(3) * 0.5 ## observer in center of simulation volume 

N_skymaps_near = 10  ## number of skymaps taken form constrained ray

seed = 42 ## seed for random number generators
f_safety = 1.25  # factor to overestimate required number of segments in order to certainly reach path length
redshift_accuracy = 4 # minimum number of 0 digits for redshift of final snapshot to be considered as z=0
average_length_segment = 0.8 ## average length of chopped segment relative to size of considered volume

N_workers_MakeNearRays = 12  ## make reasonable choice, number of maximum performance
N_workers_MakeChoppedRays = 12  ## make reasonable choice, number of maximum performance
N_workers_ReadRays = 16  ##   "-"  , based on ray distance, resolution and memory

fields = ['Density','Bx', 'By', 'Bz']   ## fields of interest to calculate DM, RM & SM



#########  dependend paramters, change with caution ########

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
redshift_skymaps = np.arange(0,redshift_max+0.1,0.1) ## redshifts of skymaps for constrained results

ts = TimeSeries( root_data + models[model_tag][0] )[::-1]  ## load series of simulation snapshots with yt
ds0 = ts[-1]  ## initial snapshot
dsz0 = ts[0]  ## final z=0 snapshot
dsz1 = ts[1]  ## pre-final snapshot

cell_width_comoving = (dsz0.domain_width.in_units('cm')/dsz0.domain_dimensions)[0]

# redshift half time between final two snapshots, after which the final snapshot is used
redshift_trans = z_from_t(  np.mean( [ t_from_z(dsz0.current_redshift), t_from_z(dsz1.current_redshift) ] ) )

# maximum redshift of constrained results ( sphere of 0.5 edgelenth of border from observer at center )
redshift_max_near = z_from_t( t_from_z( 0 ) - dsz0.domain_width.in_cgs()[0] * min( border[1] - border[0] ) / 2/ speed_of_light) # get z from time = time now minus travel time ( = distance / LightSpeed)
redshift_skymaps_near = np.linspace(0,redshift_max_near,1+N_skymaps_near)  ## redshifts of constrained skymaps


z_snaps, redshift_snapshots = RedshiftSnapshots( ts, redshift_max, redshift_max_near, redshift_trans, redshift_accuracy )

traj_length_min = min( border[1] - border[0] ) / 2                  ## minimum length of LoS segment, set to half box edgelength ( maximum length that doesn't exclude positions )
box_fractions = BoxFractions( ts, min(domain_width_code), redshift_snapshots )  ## number of volume transitions required for the path length in snapshot
N_choppers = np.ceil( np.array( box_fractions ) / average_length_segment * f_safety ).astype('int')  ## number of required segments, accounting for the average length of segments
N_choppers_total = N_choppers*npix  ## total number of required segments
