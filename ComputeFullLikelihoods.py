import numpy as np, sys
from PreFRBLE.likelihood import ComputeFullLikelihood, ComputeTelescopeLikelihood, LikelihoodTelescope
from PreFRBLE.parameter import telescopes, populations

scenario = {
    'IGM' : ['primordial'],
    'Host' : ['Rodrigues18'],
#    'Inter' : ['Rodrigues18'],
    'Local' : ['Piro18/wind'],
}
'''
scenario = {
    'IGM' : ['primordial'],
    'Host': ['Heesen11/dirty'],
    'Local': ['Piro18/wind'], 
}
'''
## force new computation of existing results
force=True

models_IGMF = [ 'alpha%i-3rd' % i for i in range(1,10) ]
f_IGMs = np.arange(0.3,0.91,0.1)


## first, test with limited set of IGMF models
#models_IGMF = [models_IGMF[i] for i in [0,2,-1]]

#'''
## first measures that do not depend on B (only one IGM model needed)
model = 'primordial'

''' DONE for host
## tau only for primordial, since tau_IGM << tau, hence not sensible to changes in f_IGM  
for telescope in telescopes:
    for population in populations:
        print( 'tau', scenario, telescope, population, file=sys.stderr)
        print( 'tau', scenario, telescope, population, file=sys.stdout)
        LikelihoodTelescope( measure='tau', telescope=telescope, population=population, force=population == populations[0] and telescope==telescope[0] and force, dev=True, **scenario )
'''

'''
for f_IGM in f_IGMs:
    tmp = scenario.copy()
    tmp['IGM'] = ["%s_C%.0f" % (model, 1000*f_IGM) ]
    for telescope in telescopes:
        for population in populations:
            print( 'DM', tmp, telescope, population, file=sys.stderr)
            LikelihoodTelescope( measure='DM', telescope=telescope, population=population, force=population == populations[0] and telescope==telescope[0] and force, dev=True, **tmp )
'''


## then measures that depend on B

if len(sys.argv) != 2:
    raise ValueError( "Please provide number of IGMF model [0-8]" )

i=int(sys.argv[1])
#j=int(sys.argv[2])

for model in models_IGMF[i:i+1]:
    for f_IGM in f_IGMs:
        tmp = scenario.copy()
        tmp['IGM'] = ["%s_C%.0f" % (model, 1000*f_IGM) ]
#        print( tmp )
#        ComputeTelescopeLikelihood( scenario=scenario, force=force)
        for telescope in telescopes:
            for population in populations:
                print( 'RM', tmp, telescope, population, file=sys.stderr)
                print( 'RM', tmp, telescope, population, file=sys.stdout)
                print( 'force', population == populations[0], telescope==telescopes[0], force )
                LikelihoodTelescope( measure='RM', telescope=telescope, population=population, force=population == populations[0] and telescope==telescopes[0] and force, dev=True, **tmp )
#'''        

