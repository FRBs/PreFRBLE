from PreFRBLE.likelihood import ComputeFullLikelihood

scenario = {
    'IGM' : ['primordial'],
    'Host' : ['Rodrigues18/smd'],
    'Inter' : ['Rodrigues18'],
    'Local' : ['Piro18/wind'],
}

## force new computation of existing results
force=False  

models_IGMF = [ 'alpha%i-3rd' % i for i in range(1,10) ]

## first, test with limited set of IGMF models
models_IGMF = [models_IGMF[i] for i in [0,2,-1]]

ComputeFullLikelihood( scenario=scenario, models_IGMF=models_IGMF, force=force, N_processes=1)


