import numpy as np, sys
from PreFRBLE.likelihood import ComputeFullLikelihood, ComputeTelescopeLikelihood, LikelihoodTelescope
from PreFRBLE.parameter import telescopes, populations


### This execution file is used to compute and write to file all likelihoods for full LoS scenarios and telescope observations
###   in case that the computing cluster does not allow to use jupyter notebooks...
###   After these calculations are done, other calculations will be finished much fast



scenario = {
    'IGM' : ['primordial'],
    'Host' : ['Rodrigues18'],
    'Inter' : ['Rodrigues18'],
    'Local' : ['Piro18/wind'],
    'N_inter' : True
}
'''
scenario = {
    'IGM' : ['primordial'],
    'Host': ['Heesen11/dirty'],
    'Local': ['Piro18/wind'], 
}
'''

scenarios = [
    {'IGM' : ['primordial'],'Host' : ['Rodrigues18'],'Local' : ['Piro18/wind'],},                                         ## no intervening galaxy
    {'IGM' : ['primordial'],'Host' : ['Rodrigues18'],'Local' : ['Piro18/wind'],'Inter' : ['Rodrigues18']},                ## certain intervening galaxy
#    {'IGM' : ['primordial'],'Host' : ['Rodrigues18'],'Local' : ['Piro18/wind'],'Inter' : ['Rodrigues18'],'N_inter':True}, ## NInter intervening galaxies (realistic estimate for unknown interveners)
]



## force new computation of existing results for full likelihood at different redshifts
force=True  ## should be True in case old likelihoods shall be replaced, can be set to False if computation of new likelihoods crashes, in order to not compute results twice

models_IGMF = [ 'alpha%i-3rd' % i for i in range(1,10) ]
f_IGMs = np.arange(0.3,1.01,0.1) #[-1:]  ##              !!!!!!!!!


## start with measures that do not depend on B (only one IGM model needed)
model = 'primordial'

''' 
## tau only for primordial with f_IGM=1, since tau_IGM << tau, hence not sensible to changes in f_IGM. However, should be considered for smaller choices of outer scale in the IGM than 50 kpc
for scenario_tmp in scenarios:
    for telescope in telescopes:
        for population in populations:
            print( 'tau', scenario_tmp, telescope, population, file=sys.stdout)
            print( 'tau', scenario_tmp, telescope, population, file=sys.stderr)
            LikelihoodTelescope( measure='tau', telescope=telescope, population=population, force=population == populations[0] and telescope==telescopes[0] and force, dev=True, progres_bar=True, **scenario_tmp)
#'''

#'''
for f_IGM in f_IGMs:
    tmp = scenario.copy()
    if f_IGM < 1:
        tmp['IGM'] = ["%s_C%.0f" % (model, 1000*f_IGM) ]
    for telescope in telescopes:
        for population in populations:
            print( 'DM', tmp, telescope, population, file=sys.stdout)
            print( 'DM', tmp, telescope, population, file=sys.stderr)
            LikelihoodTelescope( measure='DM', telescope=telescope, population=population, force=population == populations[0] and telescope==telescopes[0] and force, dev=True, progres_bar=True, **tmp )


#'''

#'''

## then measures that depend on B

#if len(sys.argv) != 2:
#    raise ValueError( "Please provide number of IGMF model [0-8]" )

#i=int(sys.argv[1])

i=8

#for model in models_IGMF[i:i+1]:
for model in models_IGMF:
    for f_IGM in f_IGMs[6:7]:
        tmp = scenario.copy()
        if f_IGM < 1:
            tmp['IGM'] = ["%s_C%.0f" % (model, 1000*f_IGM) ]
        else:
            tmp['IGM'] = [model]
        for telescope in telescopes:
            for population in populations:
                print( 'RM', tmp, telescope, population, file=sys.stdout)
                print( 'RM', tmp, telescope, population, file=sys.stderr)
                LikelihoodTelescope( measure='RM', telescope=telescope, population=population, force=population == populations[0] and telescope==telescopes[0] and force, dev=True, progres_bar=True, **tmp )
#'''        

