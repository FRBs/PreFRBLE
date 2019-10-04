'''
Produce Probability functions of LoS observables for sources in the distant universe, outside the constrained volume
several snapshots of the simulation are combined using cosmological data stacking (e.g., da Silva et al. 2000)
Note that the light-travel distance between snapshots exceeds the simulation volume (Vazza-Hackstein models), hence LoS to high redshift are obtained by stacking segments, distributed randomly troughout the volume

'''

from Plots import PlotLikelihoods
from Rays import CreateLoSSegments, CreateLoSsObservables
from LikelihoodFunctions import MakeFarLikelihoodFunction
from multiprocessing import Pool
from time import time
t0 = time()

models=['primordial', 'astrophysical']  ## models to be considered for the magnetic field, provided as B~rho relations in relations_file

models = ['primordial', 'B9b', 'B9.5b', 'B10.0b', 'B10.5b', 'B11b', 'B13b', 'B15b', 'B17b' ]

## parameters for the likelihood fuction
bins = 100  ## number of bins for the likelihood functions
DM_range = [ 1e1, 1e4 ]  ## range for the likelihood function of DM
SM_range = [ 1e-8, 1e0 ]  ## range for the likelihood function of DM
RM_range = [ 1e-6, 1e3 ] ## range for the likelihood function of |RM|




## read raw data of cells along segments that form the LoS and write them to temporary ray files
### these temp files can take up huge amounts of space, before they are reduced to their observables. For a huge number of LoS, repeat the first to steps - extraction and reduction - with a smaller number of LoS that your system can digest. It is also advised to perform this lengthy computation in parallel on several nodes by calling this file with different 'off'. If you know a more elegant version, let me know! - shackste

##   technical parameters
span_tot = 24   ## number of LoS to be computed
off = 8          ## index number of the first LoS
N_workers = 4   ## number of processes to work in parallel

span=span_tot/N_workers ##  size of bunch to be computed by single process
r = [ range(i+off, i+span+off) for i in range(0,span_tot,span) ]  ## bunches of LoS indices to be produced by the individual process

## extract raw data along segments that form LoS
pool = Pool(N_workers)
#pool.map( CreateLoSSegments, r  )
#map( CreateLoSSegments, r  )
pool.close()
pool.join()


## combine segments to full LoS and reduce raw data to LoS observables, results saved to LoS_observables_file
#CreateLoSsObservables( models=models, remove=True )
#CreateLoSsObservables( models=models, remove=False )

### repeat the steps above until you reach a decent amount of LoS reduced to observables in LoS_observables_file
### then continue with the functions below to obtain the likelihood function



## compute the likelihood function as probability density of observables from LoS and write to probability_file_IGM
## compute likelihood functions of DM and |RM|, only the latter differ between magnetic field models
bunch = 128  ## bunch size for internal computation, too big crashes memory
#P = MakeFarLikelihoodFunction( bins, DM_range, measure='DM', absolute=False, bunch=bunch, model='primordial' )
PlotLikelihoods( measure='DM', model='primordial', typ='far', plot_every=1 )
#P = MakeFarLikelihoodFunction( bins, SM_range, measure='SM', absolute=False, bunch=bunch, model='primordial' )
PlotLikelihoods( measure='SM', model='primordial', typ='far', plot_every=1 )
for model in models:
    print model
#    P = MakeFarLikelihoodFunction( bins, RM_range, measure='RM', absolute=True, bunch=bunch, model=model )
    PlotLikelihoods( measure='RM', model=model, typ='far', absolute=True, plot_every=1 )
print "This took %.0f seconds" % (time()-t0)



