import numpy as np

## sample distributions

def RandomSample( N=1, P=np.array(0), x=np.array(0), log=True ):
    ### return sample of N according to likelihood function P(x) 
    ###  P is renormalized probability density, i. e. sum(P*dx)=1
    ###  log: indicates whether x is log-scaled
    Pd = P*np.diff(x)
    if np.round( np.sum(Pd), 4) != 1:
        sys.exit( " 1 != %f" % (np.sum(Pd)) )
    f = Pd.max()
    lo, hi = x[0], x[-1]
    if log:
        lo, hi = np.log10( [lo,hi] )
    res = []
    while len(res) < N:
        r = np.random.uniform( high=hi, low=lo, size=N )
        if log:
            r = 10.**r
        z = np.random.uniform( size=N )
        p = Likelihoods( r, P/f, x )
        res.extend( r[ np.where( z < p )[0] ] )
    return res[:N]


def FakeFRBs( measures=['DM','RM'], N=50, telescope='CHIME', population='SMD', **scenario):
    ### returns measures of a fake survey of N FRBs expected to be observed by telescope assuming population & scenario for LoS
    FRBs = {}
    for measure in measures:
        ## load likelihood function 
        if measure == 'RM':
            ## due to the ionosphere foreground, only allow for RM > 1 rad m^-2 to be observed
            P, x = LikelihoodMeasureable( min=RM_min, measure=measure, telescope=telescope, population=population, **scenario )
        else:
            P, x = GetLikelihood_Telescope( measure=measure, telescope=telescope, population=population, **scenario )
    
        ##   sample likelihood function
        FRBs[measure] = RandomSample( N, P, x )
    
    return FRBs

def uniform_log( lo=1., hi=2., N=10 ):
    ## returns N samples of a log-flat distribution from lo to hi
    lo = np.log10(lo)
    hi = np.log10(hi)
    return 10.**np.random.uniform( lo, hi, N )


