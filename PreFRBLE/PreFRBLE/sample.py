import sys, numpy as np
from PreFRBLE.likelihood import LikelihoodMeasureable, GetLikelihood_Full, GetLikelihood_Telescope, GetLikelihood_Redshift, RandomSample
from PreFRBLE.physics import measure_range




############################################################################
######################## SAMPLE DISTRIBUTIONS ##############################
############################################################################


def SampleLogFlat( lo=1., hi=2., N=10 ):
    """ returns an N-sample of a log-flat distribution from lo to hi """
    lo = np.log10(lo)
    hi = np.log10(hi)
    return 10.**np.random.uniform( lo, hi, N )





def FakeFRBs( measures=['DM','RM'], N=50, telescope='CHIME', population='SMD', measureable=True, **scenario):
    """ returns measures of a fake survey of N FRBs expected to be observed by telescope assuming population & scenario for LoS

    optional: reduce range to measureable values and renormalize to 1 (should be used e. g. to reproduce a set of FRB with observed RM )

    """
    FRBs = { 'redshift':np.array([])}
    for m in measures:
        FRBs[m] = np.array([])
    ## determine how many FRBs expected per redshift bin
    P, zs = GetLikelihood_Redshift( population=population, telescope=telescope )

#    P = 1./(zs[-1]-zs[0])  ### !!! REMOVE, used to test validity of results at all redshifts

    ## size of sample per redshift bin
    N_z = np.round( N*P*np.diff(zs) ).astype('i')

    ## sample measures for each bin
    for redshift, Nz in zip( zs[1:], N_z ):
        FRBs['redshift'] = np.append( FRBs['redshift'], np.round( redshift*np.ones(Nz), 2 ) )
        for measure in measures:
            P, x = GetLikelihood_Full( measure=measure, redshift=redshift, **scenario )
            if measureable:
                P, x = LikelihoodMeasureable( P=P, x=x, min=measure_range[measure][0], max=measure_range[measure][1] )
            FRBs[measure] = np.append( FRBs[measure], RandomSample( Nz, P, x ) )
    
    return FRBs



