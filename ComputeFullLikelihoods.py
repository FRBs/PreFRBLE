import numpy as np
from PreFRBLE.likelihood import ComputeFullLikelihood, ComputeTelescopeLikelihood


scenario = {
    'IGM' : ['primordial'],
    'Host' : ['Rodrigues18/smd'],
    'Inter' : ['Rodrigues18'],
    'Local' : ['Piro18/wind'],
}

scenario = {
    'IGM' : ['primordial'],
    'Host': ['Heesen11/dirty'],
    'Local': ['Piro18/wind'], 
}

## force new computation of existing results
force=True  

models_IGMF = [ 'alpha%i-3rd' % i for i in range(1,10) ]
f_IGMs = np.arange(0.2,0.9,0.1)


## first, test with limited set of IGMF models
#models_IGMF = [models_IGMF[i] for i in [0,2,-1]]

for model in models_IGMF:
    for f_IGM in f_IGMs:
        tmp = scenario.copy()
        tmp['IGM'] = ["%s_C%.0f" % (model, 1000*f_IGM) ]
        print( tmp )
        ComputeTelescopeLikelihood( scenario=scenario, force=force)


