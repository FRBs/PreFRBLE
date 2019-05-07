from pathway import *
import numpy as np


## file roots   ### !!!! how to handle the file structure?

rays_file = root_FRB + 'rays.h5'
DMRMrays_file = root_FRB + 'DMRMrays.h5'
skymap_file = root_FRB + 'DMRMmap.h5'
probability_file = root_FRB + 'DMRMprobability_IGM.h5'
relation_file = root_data+"B_renorm_%s.txt" 

models = {
    'primordial': [ 'primordial',
                    root_data+'primordial/param.enzo',
                    False,
                    np.array([[0.25]*3,[0.75]*3]),
                    60.,
    ],
    'test': [ 'std_Buni_noVel_noG',                                    ## model (dir)
              root_data+'std_Buni_noVel_noG/gas_plus_dm_amr_adia.enzo',  ## param.enzo
              True,                                                     ## periodic
              np.array([[0.,0.,0.],[1.,1.,1.]]),                        ## borders
              99.,                                                      ## initial redshift
    ],
    'test_border': [ 'std_Buni_noVel_noG',                                    ## model (dir)
                     root_data+'std_Buni_noVel_noG/gas_plus_dm_amr_adia.enzo',  ## param.enzo
                     False,                                                     ## periodic
                     np.array([[0.2,0.2,0.2],[0.8,0.8,0.8]]),                        ## borders
                     99.,                                                      ## initial redshift of MHD simulation
    ],
    'test_uniform': [ 'std_Buni_noVel_noG',                                    ## model (dir)
                     root_data+'std_Buni_noVel_noG/gas_plus_dm_amr_adia.enzo',  ## param.enzo
                     False,                                                     ## periodic
                     np.array([[0.2,0.2,0.2],[0.8,0.8,0.8]]),                        ## borders
                     99.,                                                      ## initial redshift
    ],
}
