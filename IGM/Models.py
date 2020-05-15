'''
specific parameters for the IGM models

'''


from pathway import *
import numpy as np


## file roots   ### !!!! how to handle the file structure?

rays_file = root_FRB + 'rays.h5'
LoS_observables_file = root_FRB + 'LoS_observables.h5'
skymap_file = root_FRB + 'observables_map.h5'
likelihood_file_IGM = root_FRB + 'observables_likelihood_IGM.h5'
relation_file = root_data+"B_renorm_%s.txt" 

models = {
    'primordial': [ 'primordial',                       ## model (name of folder)
                    root_data+'primordial/param.enzo',  ## ENZO parametet file
                    False,                              ## periodic
                    np.array([[0.25]*3,[0.75]*3]),      ## borders
                    60.,                                ## initial redshift
                ]
}
