'''
Produce Probability functions of LoS observables within the constrained volume

'''


import matplotlib
matplotlib.use('Agg')  ## to plot file without display

from Rays import MakeNearRays, CollectRays
from Skymaps import MakeNearSkymaps
from LikelihoodFunctions import MakeNearLikelihoodFunction
from Plots import PlotSkymaps

import time, sys
t0 = time.time()


models=['primordial', 'astrophysical'] 

bins = 100                  ## number of bins for the likelihood function
RM_range = (-200,200)       ## range of likelihood function of RM
RM_range_log = (1e-6,1e2)   ## range of likelihood function of log( |RM| )


## from IGM simulation, read raw data of cells along the LoS and collect them to rays_file
MakeNearRays()

#sys.exit()
CollectRays( remove=True )

## compute observables from raw data and save it to full-sky maps in skymap_file
MakeNearSkymaps( models=models )


## compute the likelihood functions as probability density of the full-sky data of an observable and save them to probability_file_IGM
## compute likelihood functions of log(DM), RM and log(|RM|), only the latter differ between magnetic field models
MakeNearLikelihoodFunction( measure='DM', bins=bins )# , range=(1e-1,3e3) )
MakeNearLikelihoodFunction( measure='SM', bins=bins )# , range=(1e-1,3e3) )
for model in models:
    MakeNearLikelihoodFunction( measure='RM', absolute=True, bins=bins, model=model, range=RM_range_log )
    MakeNearLikelihoodFunction( measure='RM', absolute=False, bins=bins, model=model, range=RM_range )



## Plot full-sky maps of DM and |RM|, the latter for all models
RM_min, RM_max = 3e-11, 6.5  ## value range for full-sky maps
PlotSkymaps( measure='DM', min=0.75, max=1350. )
PlotSkymaps( measure='SM' ) #, min=0.75, max=1350. )
for model in models:
    PlotSkymaps( measure='RM', model=model, min=RM_min, max=RM_max )


print 'finished in %.0f seconds' % (time.time()-t0)
