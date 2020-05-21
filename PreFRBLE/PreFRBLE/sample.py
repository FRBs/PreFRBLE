import sys, numpy as np
from PreFRBLE.likelihood import GetLikelihood
from PreFRBLE.Scenario import Scenario



############################################################################
######################## SAMPLE DISTRIBUTIONS ##############################
############################################################################


def SampleLogFlat( lo=1., hi=2., N=10 ):
    """ returns an N-sample of a log-flat distribution from lo to hi """
    lo = np.log10(lo)
    hi = np.log10(hi)
    return 10.**np.random.uniform( lo, hi, N )





def FakeFRBs( measures=['DM','RM'], N=50, measureable=True, smooth=True, scenario=False):
    """ 
    returns measures of a fake survey of N FRBs expected to be observed by telescope assuming population & scenario for LoS

    optional: reduce range to measureable values and renormalize to 1 (should be used e. g. to reproduce a set of FRB with reasonable RM )

    """

    ## prepare result
    FRBs = { 'redshift':np.array([])}
    for m in measures:
        FRBs[m] = np.array([])
    ## determine how many FRBs expected per redshift bin
    redshift_scenario = Scenario( **scenario.Properties( regions=False ) )
    Lz = GetLikelihood( 'z', redshift_scenario )

    ## size of sample per redshift bin
    N_z = np.round( N*Lz.Probability() ).astype('i')


    tmp = Scenario( redshift=1.0, **scenario.Properties( identifier=False ) ) 
    ## sample measures for each bin
    for redshift, Nz in zip( Lz.x[1:], N_z ):
        tmp.redshift = redshift
        FRBs['redshift'] = np.append( FRBs['redshift'], np.round( redshift*np.ones(Nz), 2 ) )  ### ugly, since redshifts are quantized, replace by uniform random
        for measure in measures:
            L = GetLikelihood( measure, tmp )
            if smooth:
                L.Smooth()
            if measureable:
                L.Measureable()
            FRBs[measure] = np.append( FRBs[measure], L.RandomSample( Nz ) )
    

    return FRBs



