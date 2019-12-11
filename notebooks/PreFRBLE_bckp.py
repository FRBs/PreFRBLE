import numpy as np
from Convenience import *

zs_ = np.arange(0.1,6.1,0.1)  ## redshift bins !!! more clean please 




def LikelihoodRedshift( DMs, model, SMs=None, population='flat', telescope='None' ):
    ## returns likelihood function of redshift for an observed DM (and SM, optionally)
    ## can be used to obtain estimate and deviation
    

    Pz = np.zeros( [len(DMs),len(zs_)] )
    ## for each redshift
    for iz, z in enumerate(zs_):
        ## calculate the likelihood of observed DM 
        Pz[:,iz] = Likelihoods( DMs, *FullProbability( typ='DM', z=z, density=True, **model) ) 
    
    ## optionally, improve redshift estimate with additional information from SM, which is more sensitive to high overdensities in the LoS
    if SMs is not None:
        Pz_ = np.zeros( [len(DMs),len(zs_)] )
        ## for each redshift
        for iz, z in enumerate(zs_):
            ## calculate the likelihood of observed SM
            Pz_[:,iz] = Likelihoods( SMs, *FullProbability( typ='SM', z=z, density=True, **model) ) 
        Pz *= Pz_
    
    ## consider prior likelihood on redshift according to FRB population and telescope selection effects 
    if population == 'flat':
        pi_z = np.array([1.])
    else:
        pi_z = Likelihoods_Redshift( population=population, telescope=telescope )
    Pz *= np.resize( pi_z, [1,len(zs_)] )
                    
    ## renormalize to 1 for every measure
    Pz /= np.resize( np.sum( Pz * np.resize( np.diff( zs_range ), [1,len(zs_range)-1] ), axis=1 ), [len(DMs),1] )  
    
    return Pz, zs_range



def CombinedLikelihood( DMs, RMs, scenario={}, prior_BO=1., SMs=None, population='flat', telescope='None' ):
    ## returns the likelihood for a scenario to produce tuples of DMs and RMs (and SMs, optionally)
    ## allows to consider different population of FRBs as well as selection effects by the telescope
    
    res = np.zeros( len(DMs) ) ## container for final results
    
    #### !!!!! careful here!!! the full probability space is not normalized to 1 !!!!
    
    ## first, obtain likelihood function of source redshift from DM (and SM, optionally)
    P_z_DMs, zs = LikelihoodRedshift( DMs, scenario, SMs=SMs, population=population, telescope=telescope )
    
    ## for each redshift
    for z, p_z in zip( zs_, P_z_DMs.transpose() ):
        ## add likelihood to produce observed RM, multiplied by likelihod on redshift and prior on IGMF model
        res += prior_BO * p_z * Likelihoods( RMs, *ObservableProbability( typ='RM', z=z, density=False, **scenario) )
    return res

def CombinedBayesFactor( DMs, RMs, scenario1, scenario2, SMs=None, population='flat', telescope='None' ):
    ## returns the total Bayes factor, the corroboration towards scenario1 over scenario2, based on observed tuples of DMs and RMs (and SMs, optionally) 
    ## always use Bayes factor to compare likelihoods of scenarios (since likelihoods of numerous measurements -> 0 )
    return np.prod( CombinedLikelihood( DMs, RMs, scenario1, SMs=SMs, population=population, telescope=telescope ) / CombinedLikelihood( DMs, RMs, scenario2, SMs=SMs, population=population, telescope=telescope ) )
