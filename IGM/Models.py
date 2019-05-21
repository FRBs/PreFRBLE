'''
specific parameters for the IGM models

'''


from pathway import *
import numpy as np


## file roots   ### !!!! how to handle the file structure?

rays_file = root_FRB + 'rays.h5'
DMRMrays_file = root_FRB + 'DMRMrays.h5'
skymap_file = root_FRB + 'DMRMmap.h5'
probability_file_IGM = root_FRB + 'DMRMprobability_IGM.h5'
relation_file = root_data+"B_renorm_%s.txt" 

models = {
    'primordial': [ 'primordial',                       ## model (name of folder)
                    root_data+'primordial/param.enzo',  ## ENZO parametet file
                    False,                              ## periodic
                    np.array([[0.25]*3,[0.75]*3]),      ## borders
                    60.,                                ## initial redshift
}
