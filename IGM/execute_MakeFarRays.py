'''
Produce Probability functions of LoS observables for sources in the distant universe, outside the constrained volume
several snapshots of the simulation are combined using cosmological data stacking (e.g., da Silva et al. 2000)
Note that the light-travel distance between snapshots exceeds the simulation volume (Vazza-Hackstein models), hence LoS to high redshift are obtained by stacking segments, distributed randomly troughout the volume

'''

from Rays import CreateLoSSegments
from multiprocessing import Pool
import sys


## read raw data of cells along segments that form the LoS and write them to temporary ray files
### these temp files can take up huge amounts of space, before they are reduced to their observables. For a huge number of LoS, repeat the first to steps - extraction and reduction - with a smaller number of LoS that your system can digest. It is also advised to perform this lengthy computation in parallel on several nodes by calling this file with different 'off'. If you know a more elegant version, let me know! - shackste

##   technical parameters
if len(sys.argv) != 4:
    sys.exit("usage: ipython execute_MakeFarRays.py <number-of-LoS> <start-index> <number-processes/node> \n number-of-LoS should be multiple of process number" )
span_tot = int(sys.argv[1])   ## number of LoS to be computed
off = int(sys.argv[2])         ## index number of the first LoS
N_workers = int(sys.argv[3])   ## number of processes to work in parallel


span=span_tot/N_workers ##  size of bunch to be computed by single process
r = [ range(i+off, i+span+off) for i in range(0,span_tot,span) ]  ## bunches of LoS indices to be produced by the individual process

## extract raw data along segments that form LoS
pool = Pool(N_workers)
pool.map( CreateLoSSegments, r  )
#map( CreateLoSSegments, r  )
pool.close()
pool.join()



