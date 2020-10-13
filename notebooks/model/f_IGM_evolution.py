import numpy as np
import Pshirkov16
from PreFRBLE.physics import redshift_bins
from PreFRBLE.likelihood import LikelihoodFunction, Scenario


## physical parameters
B0 = 1e-3 # muG        ## magnetic field strength at n=1.8e-7
lambda_c = 10 # Mpc    ## correlation length
z_min, z_max = 0.01, 6 ## minimum, maximum redshift (z_min=0 results in numerical error)
N_LoS = 49152          ## number of drawn LoS  (equals number chosen for simulation)
bins=100

## standard values to match simulation
f_IGM0_init, f_IGM1_init = 1., 1.



## these values will be probed
f_IGM0s = np.arange(0.3,1.0,0.1)

## together with this value
f_IGM1 = 0.9

def MonteCarlo_DM_at_redshift( z, LoS ):
    """ obtain likelihood of DM for source at redshift z in scenario defined in LoS """
    DM = LoS.DM( N_LoS, z)
    scenario = Scenario( IGM=["Pshirkov16"], redshift=z, f_IGM=LoS.f_IGM0 )

    L = LikelihoodFunction(measure='DM', scenario=scenario )
    L.Likelihood( DM, log=True, bins=bins )
    return L

def MonteCarlo_DM( LoS, write=False ):
    """ obtain likelihood for sources at increasing redshift """
    Ls = []
    for z in redshift_bins[:3]:  ## !!!!
        
        L = MonteCarlo_DM_at_redshift( z, LoS )
        if write:
            print( 'write', L.scenario.Key( measure='DM' ) )
            L.Write()
        Ls.append(L)
    return Ls

def MonteCarlo_fIGMs():
    """ obtain likelihood for DM choosing different values of f_IGM at z=0 """

    ## setup physics
    LoS = Pshirkov16.Sightline( B0, lambda_c, z_max, f_IGM0=f_IGM0_init, f_IGM1=f_IGM1_init )
    
    LoS.f_IGM1 = f_IGM1
    for f_IGM0 in f_IGM0s:
        LoS.f_IGM0 = f_IGM0
        MonteCarlo_DM( LoS, write=True )

MonteCarlo_fIGMs()
