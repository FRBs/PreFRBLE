import numpy as np
from PreFRBLE.likelihood import LikelihoodMeasureable, GetLikelihood_Full, GetLikelihood_Telescope, GetLikelihood_Redshift, Likelihoods
from PreFRBLE.physics import measure_range


## sample distributions

def RandomSample( N=1, P=np.array(0), x=np.array(0), log=True ):
    ### return sample of N according to likelihood function P(x) 
    ###  P is renormalized probability density, i. e. sum(P*dx)=1
    ###  log: indicates whether x is log-scaled
    Pd = P*np.diff(x)
    if np.round( np.sum(Pd), 4) != 1:
        sys.exit( "function is not normalized, %f != 1" % (np.sum(Pd)) )
    f = Pd.max()
    lo, hi = x[0], x[-1]
    if log:
        lo, hi = np.log10( [lo,hi] )
    res = []
    while len(res) < N:
        ## create random uniform sample in the desired range
        r = np.random.uniform( high=hi, low=lo, size=N )
        if log:
            r = 10.**r
        ## randomly reject candiates with chance = 1 - P to recreate P
        z = np.random.uniform( size=N )
        p = Likelihoods( r, P/f, x )
        res.extend( r[ np.where( z < p )[0] ] )
    return res[:N]



def FakeFRBs( measures=['DM','RM'], N=50, telescope='CHIME', population='SMD', measureable=True, **scenario):
    ### returns measures of a fake survey of N FRBs expected to be observed by telescope assuming population & scenario for LoS
    FRBs = { 'redshift':np.array([])}
    for m in measures:
        FRBs[m] = np.array([])
    ## determine how many FRBs expected per redshift bin
    P, zs = GetLikelihood_Redshift( population=population, telescope=telescope )

#    P = 1./(zs[-1]-zs[0])  ### !!! REMOVE, used to test validity of results at all redshifts

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


### !!! MEASURES DON'T CARE FOR SOURCE REDSHIFT, WORTHLESS !!! ###
def FakeFRBs_old( measures=['DM','RM'], N=50, telescope='CHIME', population='SMD', **scenario):
    ### returns measures of a fake survey of N FRBs expected to be observed by telescope assuming population & scenario for LoS


    FRBs = {}
    for measure in measures:
        ## load likelihood function 
        if measure == 'RM':
            ## due to the ionosphere foreground, only allow for RM > 1 rad m^-2 to be observed
            P, x = GetLikelihood_Telescope( telescope=telescope, population=population, **scenario )
            P, x = LikelihoodMeasureable( x=x, P=P, min=RM_min )
#            P, x = LikelihoodMeasureable( min=RM_min, measure=measure, telescope=telescope, population=population, **scenario )
        else:
            P, x = GetLikelihood_Telescope( measure=measure, telescope=telescope, population=population, **scenario )
    
        ##   sample likelihood function
        FRBs[measure] = RandomSample( N, P, x )
    
    return FRBs

def uniform_log( lo=1., hi=2., N=10 ):
    ## returns an N-sample of a log-flat distribution from lo to hi
    lo = np.log10(lo)
    hi = np.log10(hi)
    return 10.**np.random.uniform( lo, hi, N )


