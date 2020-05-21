import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import trange
from sys import exit
from PreFRBLE.convenience import *
from PreFRBLE.parameter import *
from PreFRBLE.physics import *
from PreFRBLE.LikelihoodFunction import LikelihoodFunction
from PreFRBLE.Scenario import Scenario

############################################################################
############### MATHEMATICAL LIKELIHOOD STANDARD OPERATIONS ################
############################################################################

### !!! depreceated, remove
def Likelihood( data=np.arange(1,3), bins=10, range=None, density=True, log=False, weights=None, **kwargs ):
    """ wrapper for numpy.histogram that allows for log-scaled probability density function, used to compute likelihood function """
    if log:
        if range is not None:
            range = np.log10(range)
        h, x = np.histogram( np.log10(np.abs(data)), bins=bins, range=range, weights=weights )
        x = 10.**x
        h = h.astype('float64')
        if density:
            h = h / ( np.sum( h )*np.diff(x) )
    else:
        if range is None:
            range = ( np.min(data), np.max(data) )
        h, x = np.histogram( data, bins=bins, range=range, density=density, weights=weights )
    
    L = LikelihoodFunction( P=h, x=x, **kwargs )
    return L
#    return h, x

Histogram = Likelihood ## old name, replace everywhere



### !!! depreceated, remove
def LikelihoodSmooth( P=[], x=[], dev=[], mode='MovingAverage' ):
    """ 
    Smooth likelihood function P(x)
    
    modes available:
        MovingAverage : smooth using moving average over 5 neighbouring boxes
    
    
    """
    
    norm = LikelihoodNorm( P=P, x=x, dev=dev )
    
    if mode == 'MovingAverage':
        box_pts = 5
        P = np.convolve( P, np.ones(box_pts)/box_pts, mode='same' )
        
    ## smoothing doesn't conserve normalization
    P *= norm/LikelihoodNorm( P=P, x=x, dev=dev )
    
    res = [P, x]
    if len(dev)>0:
        res.append(dev)
    return res
        


### !!! depreceated, remove
def LikelihoodNorm( P=[], x=[], dev=[] ):
    """ Compute norm of likelihood function P """
    return np.sum(P*np.diff(x))


### !!! depreceated, remove
def LikelihoodDeviation( P=[], x=[], N=1 ):
    """ compute relative deviation (Poisson noise) of likelihood function of individual model obtained from sample of N events """
    res =  ( P*np.diff(x)*N )**-0.5
    res[ np.isinf(res) + np.isnan(res)] = 0
    return res



### !!! depreceated, remove
def Likelihoods( measurements=[], P=[], x=[], dev=[], minimal_likelihood=0., density=False ):
    """
    returns likelihoods for given measurements according to likelihood function given by P and x


    Parameters
    ---------
    measurements : array_like
        measurements for which the likelihood shall be returned
    P : array_like, shape(N)
        likelihood function
    x : array_like, shape(N+1)
        range of bins in likelihood function
    dev : array_like, shape(N), optional
        deviation of likelihood function, if given, return deviation of returned likelihoods
    minimal_likelihood : float
        value returned in case that measurement is outside x
    density : boolean
        if True, return probability density ( P ) instead of probability ( P*dx )

    Returns
    -------
    likelihoods: numpy array, shape( len(measurements) )
        likelihood of measurements = value of P*dx for bin, where measurement is found
    """



    likelihoods = np.zeros( len( measurements ) ) ## collector for likelihoods of measurements
    deviations = likelihoods.copy()
    prob = P if density else P*np.diff(x)  ## probability for obtaining measure from within bin
    isort = np.argsort( measurements )   ## sorted order of measurements
    i = 0  ## marker for current bin
    ## for each measurement (in ascending order)
    for m, i_s in zip( np.array(measurements)[isort], isort ):
    ##   check bins >= previous results
        for xi in x[i:]:
    ##      whether measure is inside
            if m >= xi:  ## measure is bigger than current bin range
                ##   set marker and continue with next bin
                i += 1   
                continue
            else:        ## otherwise, measure is in the bin
                ## put result in correct place and stop checking bins
                likelihoods[i_s] = prob[i-1]  if i > 0 else minimal_likelihood  ## if that was the lowest bound, probability is ->zero if measurement is outside the range of P, i. e. P~0
                if len(dev):
                    deviations[i_s] = dev[i-1] if i > 0 else 1
                break    ## continue with the next measurement
        else:
            ## if measure is bigger than the last bin
            likelihoods[i_s] = minimal_likelihood  ## probability is zero if measurement is outside the range of P, i. e. P~0
            if len(dev):
                deviations[i_s] = 1
    
#    likelihoods = np.array( likelihoods )
    if len(dev):
        return likelihoods, deviations
    else:
        return likelihoods
        

### !!! depreceated, remove
def RandomSample( N=1, P=np.array(0), x=np.array(0), log=True ):
    """
    returns sample of size N according to likelihood function P(x) 

    Parameter
    ---------
    P, x : array-like
        renormalized probability density function, i. e. sum(P*np.diff(x))=1
    log: indicates whether x is log-scaled

    Output
    ------

    res : list of N values, distributed according to P(x)

    """
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
        ## obtain probability for bins where measures measures are found
        p = Likelihoods( r, P/f, x, density=False ) ### renormalize pdf to maximum value of probability, such that values at maximum probability are never rejected. This minimizes the number of rejected random draws
        res.extend( r[ np.where( z < p )[0] ] )
    return res[:N]



### !!! depreceated, remove
def LikelihoodShift( P=[], x=[], dev=[], shift=1. ):
    """ Shift x-values of likelihood function and renormalize accordingly: P'(x|shift) = shift * P(shift*x|1) """
    # x' = shift*x, thus P' = P dx/dx' = P / shift 
    res = [ P/shift, x*shift ]
    if len(dev):
        res.append(dev)
    return res



def LikelihoodsAdd( *Ls, shrink=False, weights=None, dev_weights=None, renormalize=1, min=None, max=None, smooth=True, scenario=False ):
    """
    add together several likelihood functions

    Parameters
    ----------
    Ls : array-like
        list of LikelihoodFunction objects to be added
    shrink : integer
        determine number of bins in result, otherwise use size of Ps[0]
    weights : array-like, len=Ls.shape[0], optional
        provide weights for the likelihood functions, assumed =1 if not given
    dev_weights : array-like, len=Ps.shape[0], optional
        provide deviation for the weights, used to compute error of result
    renormalize : float
        renormalization of result
    min, max : float
        indicate minimum and/or maximum value of summed function
    smooth : boolean
        if True, return smoothed L ( LikelihoodSmooth )
    scenario : Scenario-object
        scenario described by summed LikelihoodFunction

    Returns
    -------
    LikelihoodFunction : summed likelihood

    """
    
    
    if len(Ls) == 1:
        ## if only one function is given, return the original
        L = Ls[0]
        ### !!! depreceated
        if renormalize: ## maybe renormalized to new value
            L.Renormalize( renormalize ) ### !!! depreceated
        if smooth: ### !!! depreceated
            L.smooth()
        return L

    ## new function support
    l = len(Ls[0].P)
    if shrink:
        l = shrink
    x_min = min if min else np.min( [L.x.min() for L in Ls] )
    x_max = max if max else np.max( [L.x.max() for L in Ls] )
    if Ls[0].log:
        x = 10.**np.linspace( np.log10(x_min), np.log10(x_max), l+1 )
    else:
        x = np.linspace( x_min, x_max, l+1 )
        
    if weights is None:
        weights = np.ones( len(Ls) )
    if dev_weights is None:
        dev_weights = np.zeros( len(Ls) )

    P = np.zeros( l )
    dev = P.copy()

    ## for each function
    for i_L, (L, w) in enumerate( zip(Ls, weights) ):
    ##   loop through target bins
        for ib, (b0, b1) in enumerate( zip( x, x[1:] ) ):
            ## start where bins are not too low
            if b1 < L.x[0]:
                continue
            ## stop when bins become too high
            if b0 > L.x[-1]:
                break
    ##     identify contributing bins
            ix, = np.where( ( L.x[:-1] < b1 ) * ( L.x[1:] > b0 ) )
            if len(ix) == 0:
                continue   ## skip if none
            elif len(ix) == 1:
                P[ib] += w * L.P[ix]  ## add if one
                if len(L.dev)>0:
                    dev[ib] += (w * L.P[ix])**2 * ( L.dev[ix]**2 + dev_weights[i_L]**2 )
            else:  ## compute average of contributing bins
    ##     get corresponding ranges
                x_ = L.x[np.append(ix,ix[-1]+1)].copy()
    ##     restrict range to within target bin
                x_[0], x_[-1] = b0, b1
    ##     add weighed average to target likelihood
                add = w * np.sum( L.P[ix]*np.diff(x_) ) / (b1-b0)
                P[ib] += add
                if len(L.dev)>0:
                    dev[ib] += add**2 * ( np.sum( ( L.dev[ix]*L.P[ix]*np.diff(x_) )**2 ) /np.sum( ( L.P[ix]*np.diff(x_) )**2 )  + dev_weights[i_L]**2 )
    dev = np.sqrt(dev)/P
    dev[ np.isnan(dev) ] = 0
    
    L = LikelihoodFunction( P=P, x=x, dev=dev, typ=Ls[0].typ, measure=Ls[0].measure )  ### this L is still missing scenario
    
    L.Renormalize( renormalize )
    if smooth:
        L.Smooth()
        
    return L



### !!! old version, remove
def LikelihoodsAdd_old( Ps=[], xs=[], devs=[], log=True, shrink=False, weights=None, dev_weights=None, renormalize=False, min=None, max=None, smooth=True ):
    """
    add together several likelihood functions

    Parameters
    ----------
    Ps : array-like
        list of likelihood functions
    xs : array-like
        list of bin ranges of likelihood functions
    devs : array-like, optional
        list of deviations of likelihood functions, used to compute deviation of result
    log : boolean
        indicate wether xs are log-scaled
    shrink : integer
        determine number of bins in result, otherwise use size of Ps[0]
    weights : array-like, len=Ps.shape[0], optional
        provide weights for the likelihood functions
    dev_weights : array-like, len=Ps.shape[0], optional
        provide deviation for the weights
    renormalize : float, optional
        renormlization of result
    min, max : float
        indicate minimum and/or maximum value of added function
    smooth : boolean
        if True, return smoothed P ( LikelihoodSmooth )

    Returns
    -------
    P, x, (dev) : summed likelihood function values, range, (deviation)

    """

    if len(Ps) == 1:
        ## if only one function is given, return the original
        P, x = Ps[0], xs[0] 
        norm = 1
        if renormalize: ## maybe renormalized to new value
            norm = renormalize/np.sum( P*np.diff(x) )
            P *= norm
        if smooth:
            P, x = LikelihoodSmooth( P=P, x=x )

        res = [P, x]
        if len(devs) > 0:
            res.append( devs[0] )
        return res

    ## new function support
    l = len(Ps[0])
    if shrink:
        l = shrink
    x_min = min if min else np.min(xs)
    x_max = max if max else np.max(xs)
    if log:
        x = 10.**np.linspace( np.log10(x_min), np.log10(x_max), l+1 )
    else:
        x = np.linspace( x_min, x_max, l+1 )
    if weights is None:
        weights = np.ones( len(Ps) )
    if dev_weights is None:
        dev_weights = np.zeros( len(Ps) )
        
    P = np.zeros( l )
    dev = P.copy()

    ## for each function
    for i_f, (f, x_f, w) in enumerate( zip(Ps, xs, weights) ):
    ##   loop through target bins
        for ib, (b0, b1) in enumerate( zip( x, x[1:] ) ):
            ## stop when bins become too high
            if b0 > x_f[-1]:
                break
    ##     identify contributing bins
            ix, = np.where( ( x_f[:-1] < b1 ) * ( x_f[1:] > b0 ) )
            if len(ix) == 0:
                continue   ## skip if none
            elif len(ix) == 1:
                P[ib] += w * f[ix]  ## add if one
                if len(devs)>0:
                    dev[ib] += (w * f[ix])**2 * ( devs[i_f][ix]**2 + dev_weights[i_f]**2 )
            else:  ## compute average of contributing bins
    ##     get corresponding ranges
                x_ = x_f[np.append(ix,ix[-1]+1)]
    ##     restrict range to within target bin
                x_[0], x_[-1] = b0, b1
    ##     add weighed average to target likelihood
                add = w * np.sum( f[ix]*np.diff(x_) ) / (b1-b0)
                P[ib] += add
                if len(devs)>0:
                    dev[ib] += add**2 * ( np.sum( ( devs[i_f][ix]*f[ix]*np.diff(x_) )**2 ) /np.sum( ( f[ix]*np.diff(x_) )**2 )  + dev_weights[i_f]**2 ) 
    if len(devs)>0:
        dev = np.sqrt(dev)/P
        dev[ np.isnan(dev) ] = 0
    if renormalize:
        P *= renormalize/np.sum( P*np.diff(x) )
    if smooth:
        P, x, dev = LikelihoodSmooth( P=P, x=x, dev=dev )

    res = [P,x]
    if len(devs)>0:
        res.append( dev )
    return res


### !!! depreceated, remove
def LikelihoodShrink( P=np.array(0), x=np.array(0), dev=[], bins=100, log=True, renormalize=False, **kwargs_LikelihoodsAdd ):
    """ reduce number of bins in likelihood function, contains normalization """
    ### Actual work is done by LikelihoodsAdd, which adds up several P to new range with limited number of bins
    ### to shrink function, add P=0 with identical range
    devs = [dev,np.zeros(len(dev))] if len(dev) > 0 else []
    renorm = renormalize if renormalize else np.sum( P*np.diff(x) ) 
    return LikelihoodsAdd( [P, np.zeros(len(P))], [x,x], devs=devs, shrink=bins, log=log, renormalize=renorm, **kwargs_LikelihoodsAdd )



def LikelihoodsConvolve( *Ls, dev=True, N=50000, absolute=False, renormalize=False, smooth=True, shrink=False ):
    """
    compute convolution of likelihood functions P in brute force method, i. e. add samples of size N of each P

    Parameter
    ---------
    Ls : list of LikelihoodFunction objects
        insert likelihood functions as lists [P,x]
    N : integer
        size of sample to compute convolution and corresponding deviation
    dev : boolean
        indicate whether to return the relative deviation based on shot noise of sample with size N
    shrink : boolean   (depreceated)
         if True, reduce number of bins of result to standard number of bins
    absolute : boolean
        indicate whether likelihood describes absolute value (possibly negative)
        if True, allow to values to cancel out by assuming same likelihood for positive and negative values
    smooth : boolean
        if True, return smoothed P ( LikelihoodSmooth )
    renormalize : float (depreceated)
        renormalization factor of final result. False to keep normalization after convolution

    Returns
    -------
    P, x, (dev) : convolve likelihood function values, range (and relative deviation, if dev=True)

    """

    samples = []
    for L in Ls:
        norm = L.Norm()
    ##  obtain sample
        sample = L.RandomSample( N=N )
#        sample = np.array( RandomSample( N=N, P=P[0]/norm, x=P[1], log=log ) ) ### requires norm = 1. other cases are cared for later

    ## account for norm < 1, i. e. distribution only contributes to amount norm of values
        if norm != 1: ### random number of 1-norm events put to 0
            sample[np.random.rand(N) > norm] = 0  

    ## account for values to potentially cancel each other
        if absolute: ### random half of sample with negative sign
            sample[np.random.rand(N) > 0.5] *= -1  ### assume same likelihood for positive and negative values
        samples.append( sample )

    ## compute likelihood
    L = LikelihoodFunction( measure=Ls[0].measure, typ=Ls[0].typ )
    L.Likelihood( np.abs( np.sum( samples, axis=0 ) ), log=log, bins=Ls[0].P.size )
    if smooth:
        L.Smooth()
    L.ShotNoise( N=N )
    return L

### !!! depreceated, remove
def LikelihoodsConvolve_old( *Ps, dev=True, log=True, N=50000, absolute=False, renormalize=False, smooth=True, shrink=False ):
    """
    compute convolution of likelihood functions P in brute force method, i. e. add samples of size N of each P

    Parameter
    ---------
    Ps : likelihood functions
        insert likelihood functions as lists [P,x]
    N : integer
        size of sample to compute convolution and corresponding deviation
    dev : boolean
        indicate whether to return the relative deviation based on shot noise of sample with size N
    shrink : boolean   (depreceated)
         if True, reduce number of bins of result to standard number of bins
    log : boolean
         indicates whether x_f and x_g are log-scaled
    absolute : boolean
        indicate whether likelihood describes absolute value (possibly negative)
        if True, allow to values to cancel out by assuming same likelihood for positive and negative values
    smooth : boolean
        if True, return smoothed P ( LikelihoodSmooth )
    renormalize : float (depreceated)
        renormalization factor of final result. False to keep normalization after convolution

    Returns
    -------
    P, x, (dev) : convolve likelihood function values, range (and relative deviation, if dev=True)

    """
        
    samples = []
    for P in Ps:
        norm = LikelihoodNorm( *P )
    ##  obtain sample
        sample = np.array( RandomSample( N=N, P=P[0]/norm, x=P[1], log=log ) ) ### requires norm = 1. other cases are cared for later
        
    ## account for norm < 1, i. e. distribution only contributes to amount norm of values
        if norm != 1:
            sample[np.random.rand(N) > norm] = 0
            
    ## account for values to potentially cancel each other
        if absolute:
            sample[np.random.rand(N) > 0.5] *= -1  ### assume same likelihood for positive and negative values
        samples.append( sample )
    
    ## compute likelihood
    P, x = Likelihood( np.abs( np.sum( samples, axis=0 ) ), log=log, bins=len(Ps[0][0]) )
    if smooth:
        P, x = LikelihoodSmooth( P=P, x=x )
    res = [ P, x ]
    if dev:
        res.append( LikelihoodDeviation( P=P, x=x, N=N ) )
    return res


### !!! depreceated, remove
def Likelihood2Expectation( P=np.array(0), x=np.array(0), log=True,  density=True, sigma=1, std_nan=np.nan ):
    """
    computes the estimate value and deviation from likelihood function P (must be normalized to 1)


    Parameters
    --------
    P : array_like, shape(N)
        likelihood function
    x : array_like, shape(N+1)
        range of bins in likelihood function
    log : boolean
        indicates, whether x is log-scaled
    density : boolean
        indicates whether P is probability density, should always be true
    sigma : integer
        indicates the sigma range to be returned. must be contained in sigma_probability in physics.py
    std_nan
        value returned in case that P=0 everywhere. if not NaN, should reflect upper limit

    Returns
    -------
    expect: float
        expectation value of likelihood function
    deviation: numpy_array, shape(1,2)
        lower and uppper bound of sigma standard deviation width
        is given such to easily work with plt.errorbar( 1, expect, deviation )

    """
    if log:
        x_log = np.log10(x)
        x_ = x_log[:-1] + np.diff(x_log)/2
    else:
        x_ = x[:-1] + np.diff(x)/2
    ## need probability function, i. e. sum(P)=1
    if density:
        P_ = P*np.diff(x)
    else:
        P_ = P
    if np.round( np.sum( P_ ), 2) != 1:
        if np.all(P_ == 0):
            return std_nan #, [std_nan,std_nan]
        sys.exit( 'P is not normalized' )
    
    ## mean is probabilty weighted sum of possible values
    expect = np.sum( x_*P_ )
    if log:
        expect = 10.**expect

    ## exactly compute sigma range
    P_cum = np.cumsum( P_ )
    ## find where half of remaining probability 1-P(sigma) is entailed in x <= x_lo
    lo =   expect - first( zip(x, P_cum), condition= lambda x: x[1] > 0.5*(1-sigma_probability[sigma]) )[0]  
    ## find where half of remaining probability 1-P(sigma) is entailed in x >= x_hi
    hi = - expect + first( zip(x[1:], P_cum), condition= lambda x: x[1] > 1- 0.5*(1-sigma_probability[sigma]) )[0]
    
    ## if z is clearly within one bin, hi drops negative value
    #hi = np.abs(hi)

#    x_std = np.sqrt( np.sum( P_ * ( x_ - expect)**2 ) ) ### only works for gaussian, so never

    deviation = np.array([lo,hi]).reshape([2,1])

    return expect, deviation

### keep
def WeighBayesFactor( bayes=1, weight=1 ):
    """ Weigh the significance of Bayes factor bayes with weight w"""
    w_log = np.log10(weight)
    return 10.**( np.log10(bayes) * (1+np.abs(w_log))**(1 - 2*(w_log<0) - (w_log==0) )  ) 


### keep
def BayesTotalLog( bayes, axis=None ):
    """ return log10 of total bayes factor along axis """
    return np.nansum( np.log10(bayes), axis=axis)

### keep
def BayesJackknife( bayes, axis=None ):
    """ return log10 of total bayes factor and deviation from Jackknife resampling of bayes factors of individual measurements """
    mean, dev = Jackknife( bayes, BayesTotalLog, axis=axis )
    return mean, dev

### keep
def BayesFactors( P1=0, P2=0, which_NaN=False ):
    """
    compute Bayes factors = P1/P2 between two scenarios

    Parameters
    ----------
    P1, P2 : array-like
        values of LikelihoodFunction of two competing scenarios for a number of measurements
    which_NaN : boolean
        if True, print indices of likelihoods, for which Bayes factor is NaN or infinite

    Returns
    -------
    bayes : array-like
        Bayes factors

    """
    bayes =  P1/P2
    NaN = np.isnan(bayes) + np.isinf(bayes)
#    dev = np.sqrt( dev1*2 + dev2**2 )
    if np.any(NaN):
        print( "%i of %i returned NaN. Ignore in final result" %( np.sum(NaN), len(bayes) ) )
        bayes[NaN] = 1
        if which_NaN:
            which_NaN = np.where( NaN )
            print(which_NaN)
    else:
        which_NaN = None
    
    return bayes

### keep
def BayesFactor( P1=0, P2=0, dev1=0, dev2=0, which_NaN=False, axis=None ):
    """
    compute total Bayes factor = prod(P1/P2) between two scenarios

    Parameters
    ----------
    P1, P2 : array-like
        values of LikelihoodFunction of two competing scenarios for a number of measurements
    dev1/2 : array-like
        deviation of likelihoods P1/2, used to compute deviation of Bayes factor according to error propagation
    which_NaN : boolean
        if True, print indices of likelihoods, for which Bayes factor is NaN or infinite
    axis : integer
        if -1: return array of individual Bayes factor for each pair of P1 and P2
        if None: return total Bayes factor = product of individual Bayes factors
        else : return array of total Bayes factor computed along axis

    Return
    ------
    bayes, dev : array-like
        Bayes factor and deviation

    """
    NaN = True
    bayes = BayesFactors( P1=P1, P2=P2, which_NaN=NaN )

    if axis == -1:
        return bayes
    
    return BayesFactorTotal( bayes, axis=axis )



### keep
def BayesFactorTotal( bayes, mode='Jackknife', axis=None ):
    """ 
    return total bayes factor using mode

    Parameter
    ---------
    bayes : array-like
        individual bayes factors 
    mode : string
        set mode how to compute total Bayes factor (and deviation)
        'simple' : return product(bayes)
        'Jackknife' : average and deviation from Jackknife estimate

    Return
    ------
    bayes : float
        total bayes factor
    dev : float
        relative deviation of bayes

    """
    if mode == 'simple':
        return np.prod(bayes, axis=axis)
    if mode == 'Jackknife':
        bayes, dev = BayesJackknife( bayes, axis=axis )
        return 10.**bayes, dev
    


############################################################################
#################### MATHEMATICAL LIKELIHOOD OPERATIONS ####################
############################################################################

def LikelihoodRegion( region='', measure='', scenario=False ):
    """
    return likelihood for a region. if multiple models are provided, their likelihoods are summed together 

    Parameters
    ----------
    region : string
        indicate the region along line of sight
    weights: array-like, optional
         weights to be applied to the models
    smooth : boolean
        if True, return smoothed P ( LikelihoodSmooth )
    **kwargs contain extra parameters for the regions, e. g. f_IGM, L0 or N_inter
    """

    Ls = []
    for model in scenario.regions[region]:
        properties = scenario.Properties( parameter=True, identifier=True )
        properties.update( {region:model} )
        tmp = Scenario( **properties )
        L = GetLikelihood( measure=measure, scenario=tmp )
        kw = kwargs.copy()
        kw[region] = model
        scenario = Scenario( **kw )
        L = LikelihoodFunction( scenario=scenario )
        Ls.append(L)
    L_ = LikelihoodsAdd( *Ls )
    return L


def ComputeFullLikelihood( measure='', scenario=False, force=False ):
    """
    cempute and return the full likelihood function for measure in the given scenario, i. e. convolution of P from all considered regions
    result is written to likelihood_file_Full

    Parameters
    ----------
    measure : string,
        measure for which P is computed
    scenario : Scenario-object
        identifier for investigated scenario
    force : boolen
        if True: force new computation of likelihood and overwrite existing results in file

    Returns
    -------
    L : LikelihoodFunction-object
    """
    
    if not measure:
        exit( "you must provide a measure. Try: 'DM', 'RM', 'tau'" )
    if not type(scenario) == type(Scenario(redshift=0.1)):
        exit( "you must provide a reasonable Scenario" )

    ## collect likelihood functions for all regions along the LoS
    Ls = []
    for region in scenario.regions: ## for all regions considered in scenario
        if scenario.regions.get( region ):
            ## obtain the full likelihood of that region
            Ls.append( LikelihoodRegion( region=region, measure=measure, scenario=scenario ) )

    N = np.min( [N_sample[region] for region in scenario.regions.keys() ] ) ### find maximum sample size for convolution, which is minimal size of sample in all conributing likelihood functions
    L = LikelihoodsConvolve( *Ls, N=N, absolute= measure == 'RM' )

    L.scenario = scenario
    
    ## write to file
    L.Write()
    return L


### !!! depreceated, remove
def LikelihoodFull_old( measure='DM', redshift=0.1, nside_IGM=4, dev=False, N_inter=False, L0=1000., **scenario ):
    """
    return the full likelihood function for measure in the given scenario, i. e. convolution of P from all considered regions
    P, x and dev are written to likelihood_file_Full

    Parameters
    ----------
    measure : string,
        measure for which P is computed
    redshift : float,
        redshift of the source
    nside_IGM : integer,
        pixelization of IGM full-sky maps
    dev : boolean,
        indicate whether deviation of P should be returned
    N_inter : boolean
        if False: LoS should definitely entail an intervening galaxy  (P_Inter renormalized to 1)
        if True: it is unknown whether galaxies intersect the LoS or not (P_Inter renormalized to NInter(redshift) )
    L0 : float
        outer scale of turbulence in kpc assumed for IGM, default: 1000 kpc (Ryu et al.2008)
        affects tau
    **scenario : dictionary
        list of models combined to one scenario

    Returns
    -------
    P, x, (dev) : likelihood function, bin ranges, (deviation)
    """

    ## collect likelihood functions for all regions along the LoS
    Ps, xs, devs = [], [], []
    for region in regions:
        model = scenario.get( region )
        if model:
#            print('full', region, model )
#            P, x, P_dev = LikelihoodRegion( region=region, models=model, measure=measure, redshift=redshift, N_inter=N_inter, L0=L0, dev=True  )
            P = LikelihoodRegion( region=region, models=model, measure=measure, redshift=redshift, N_inter=N_inter, L0=L0, dev=True  )
#            print( region, LikelihoodNorm( P, x ) )
            Ps.append( P )
#            Ps.append( P )
#            xs.append( x )
#            devs.append( P_dev )

#            print( 'P_dev_region', region, model, P_dev.min(), P_dev.max() )

    if len(Ps) == 0:
        sys.exit( "you must provide a reasonable scenario" )
    N = np.min( [N_sample[region] for region in scenario.keys() if region in N_sample.keys()] ) ### find maximum sample size for convolution, which is minimal size of sample in all conributing likelihood functions
#    P, x, P_dev = LikelihoodsConvolve( Ps, xs, devs=devs, absolute= measure == 'RM', N=N )
    P = LikelihoodsConvolve( *Ps, N=N, absolute= measure == 'RM' )

#    print( 'P_dev_convolve', P_dev.min(), P_dev.max(), np.array(devs).max(), measure )
    
    ## write to file
    Write2h5( likelihood_file_Full, P, [ KeyFull( measure=measure, redshift=redshift, axis=axis, N_inter=N_inter, L0=L0, **scenario ) for axis in ['P', 'x', 'dev']] )

    
    if not dev:
        P = P[:-1]
    return P

def FlatPrior( measure='', x=[] ):
    """ return flat prior LikelihoodFunction object for range x """
    return LikelihoodFunction( measure=measure, P=np.ones_like(x[:-1]), x=x, dev=np.zeros_like(x[:-1]), typ='prior' )

def ComputeTelescopeLikelihood( measure='', scenario=False, force=False, progress_bar=False ):
    """
    return the likelihood function for measure expected to be observed by telescope in the given scenario
    P, x and dev are written to likelihood_file_telescope

    Parameters
    ----------
    measure : string,
        measure for which P is computed
    force : boolean,
        indicate whether full likelihood functions should be computed again (only required once per scenario)

    Returns
    -------
    L : LikelihoodFunction object
    """
    
    if not measure:
        exit( "you must provide a measure. Try: 'DM', 'RM', 'tau'" )
    if not type(scenario) == type(Scenario(redshift=0.1)):
        exit( "you must provide a reasonable Scenario" )


    ## prior on redshift is likelihood based on FRB population and telescope selection effects
    if population == 'flat':
        pi_z = FlatPrior( measure='z', x=redshift_range )
    else:
        scenario_telescope = Scenario( population=population, telescope=telescope )
        pi_z = GetLikelihood( measure='z' , scenario=scenario_telescope)

    ## possible solutions for all redshifts are summed, weighed by the prior
    Ls = []

    ## prepare scenario used for full likelihood function at increasing redshift
    tmp = scenario.copy()
    tmp.population = False
    tmp.telescope = False
    
    ## optionally, provide progress bar
    l = len(redshift_bins)
    ran = trange( l, desc='LikelihoodTelescope {} {}'.format( telescope, population ) ) if progress_bar else range( l )
    for i in ran:
        tmp.redshift = redshift_bins[i]
        L = GetLikelihood( measure=measure, scenario=tmp, force=force )
        Ls.append(L)
    L = LikelihoodsAdd( *Ls, weights=pi_z.Probability(), dev_weights=pi_z.dev )
    L.Write()
    return L


### !!! depreceated, remove
def LikelihoodTelescope_old( measure='DM', telescope='Parkes', population='SMD', nside_IGM=4, force=False, dev=False, progress_bar=False, N_inter=False, **scenario ):
    """
    return the likelihood function for measure expected to be observed by telescope in the given scenario
    P, x and dev are written to likelihood_file_telescope

    Parameters
    ----------
    measure : string,
        measure for which P is computed
    telescope : string,
        observing instrument 
    population : string,
        assumed cosmic population 
    nside_IGM : integer,
        pixelization of IGM full-sky maps
    force : boolean,
        indicate whether full likelihood functions should be computed again (only required once per scenario)
    dev : boolean,
        indicate whether deviation of P should be returned
    N_inter : boolean
        if False: LoS should definitely entail an intervening galaxy  (P_Inter renormalized to 1)
        if True: it is unknown whether galaxies intersect the LoS or not (P_Inter renormalized to NInter(redshift) )

    Returns
    -------
    P, x, (dev) : likelihood function, bin ranges, (deviation)
    """
        
    ## prior on redshift is likelihood based on FRB population and telescope selection effects 
    if population == 'flat':
        Pz = None
    else:
        Pz, zs, devz = GetLikelihood_Redshift( population=population, telescope=telescope, dev=True )
    
    ## possible solutions for all redshifts are summed, weighed by the prior
    Ps, xs, devs = [], [], []
#    for z in redshift_bins:
    ran = trange( len(redshift_bins), desc='LikelihoodTelescope {} {}'.format( telescope, population ) ) if progress_bar else range( len(redshift_bins) )
    for i in ran:
        z = redshift_bins[i]
        P, x, dev = GetLikelihood_Full( measure=measure, redshift=z, force=force, dev=True, N_inter=N_inter, **scenario )
        Ps.append(P)
        xs.append(x)
        devs.append(dev)
    P, x, dev = LikelihoodsAdd( Ps, xs, devs=devs, renormalize=1., weights=Pz*np.diff(zs), dev_weights=devz )
    Write2h5( filename=likelihood_file_telescope, datas=[P,x, dev], keys=[ KeyTelescope( measure=measure, telescope=telescope, population=population, axis=axis, N_inter=N_inter, **scenario) for axis in ['P','x', 'dev'] ] )

    res = [P,x]
    if len(dev)>0:
        res.append(dev)
    return res


### !!! depreceated, remove
def LikelihoodMeasureable( P=[], x=[], dev=[], min=None, max=None ):
    """    returns the renormalized part of full likelihood function that can be measured by telescopes, i. e. min <= x <= max """
    ## determine number of bins in result, roughly number of bins  min <= x <= max 
    bins = int(np.sum( np.prod( [x>=min if min else np.ones(len(x)), x<=max if max else np.ones(len(x)) ], axis=0 ) ))
    return LikelihoodShrink( P=P, x=x, dev=dev, min=min, max=max, renormalize=1, bins=bins, smooth=False ) ### smoothing is not reliable at border values. Here, border value is close to peak in P, hence don't smooth

    if min:
        ix, = np.where( x >= min )
        x = x[ix]
        P = P[ix[:-1]] ## remember, x is range of P, i. e. size+1
        if len(dev) > 0:
            dev = dev[ix[:-1]]
    if max:
        ix, = np.where( x <= max )
        x = x[ix]
        P = P[ix[:-1]] ## remember, x is range of P, i. e. size+1
        if len(dev) > 0:
            dev = dev[ix[:-1]]
    ## renormalize to 1
    P /= np.sum( P*np.diff(x) )
    res = [P,x]
    if len(dev) > 0:
        res.append(dev)
    return res


### !!! depreceated, remove
def LikelihoodRedshift( DMs=[], scenario={}, taus=None, population='flat', telescope='None', dev=False ):
    """
    returns likelihood functions of redshift for observed DMs (and taus)
    can be used to obtain estimate and deviation

    Parameters
    ----------
    DMs : array-like
        1D array contain extragalactic component of observed values
    taus : array-like, len(DMs), optional
        temporal smearing observed with DM 
    scenario : dictionary
        list of models combined to one scenario
    population : string
        assumed cosmic population of FRBs
    telescope: string
        instrument to observe DMs, RMs and taus
    dev : boolean
        if True, also return deviation of liklihood functions
    """

    Ps = np.zeros( [len(DMs),len(redshift_bins)] )
    devs= Ps.copy()
    ## for each redshift
    for iz, z in enumerate( redshift_bins ):
        ## calculate the likelihood of observed DM 
#        Ps[:,iz] = Likelihoods( DMs, *GetLikelihood_Full( measure='DM', redshift=z, density=True, **scenario) ) 
        Ps[:,iz], devs[:,iz] = Likelihoods( DMs, *GetLikelihood_Full( measure='DM', redshift=z, density=True, dev=True, **scenario), density=True ) ### use probability density to compare same value of DM at different redshifts. Otherwise influenced by different binning 
    
    ## improve redshift estimate with additional information from tau, which is more sensitive to high overdensities in the LoS
    ## procedure is identical, the likelihood functions are multiplied
    if taus is not None:
        Ps_ = np.zeros( [len(DMs),len(redshift_bins)] )
        devs_ = Ps_.copy()
        for iz, z in enumerate(redshift_bins):
            Ps_[:,iz], devs_[:,iz] = Likelihoods( taus, *GetLikelihood_Full( measure='tau', redshift=z, density=True, dev=True, **scenario), density=False )  ### not all tau are measureable. However, here we compare different redshifts in the same scenario, so the amount of tau above tau_min is indeed important and does not affect the likelihood of scenarios. Instead, using LikelihoodObservable here would result in wrong estimates.
        Ps *= Ps_
        devs = np.sqrt( devs**2 + devs_**2 ) 
        Ps_= 0
    
    ## consider prior likelihood on redshift according to FRB population and telescope selection effects 
    if population == 'flat':
        pi, x, pi_dev = np.array([1.]), np.arange(2), np.zeros(1)
    else:
        pi, x, pi_dev = GetLikelihood_Redshift( population=population, telescope=telescope, dev=True )
    Ps = Ps * np.resize( pi*np.diff(x), [1,len(redshift_bins)] )
    devs = np.sqrt( devs**2 + np.resize( pi_dev**2, [1,len(redshift_bins)] ) )
    ## renormalize to 1 for every DM (only if any P is not zero)
    for P in Ps:
        if np.any( P > 0):
            P /= np.sum( P*np.diff( redshift_range ) )

#    Ps = Ps / np.resize( np.sum( Ps * np.resize( np.diff( redshift_range ), [1,len(redshift_bins)] ), axis=1 ), [len(DMs),1] )
    res = [Ps, redshift_range]
    if dev:
        res.append(devs)
    return res

def LikelihoodRedshiftMeasure( measure='', data=[], scenario=False, measureable=False):
    """
    returns likelihood functions of redshift for observed data of measure, 
    can be used to obtain estimate and deviation 

    Parameters
    ----------
    measure : string
        indicate which measure is probed
    data : array-like
        1D array contain extragalactic component of observed values
    scenario : dictionary
        list of models combined to one scenario
    prior : boolean
        
    """
    
    if not measure:
        exit( "you must provide a measure. Try: 'DM', 'RM', 'tau'" )
    if scenario.redshift:
        exit( "requires scenario with telescope and population" )
    
    ## prepare scenario for increasing redshift
    tmp = scenario.copy()
    tmp.population = False
    tmp.telescope = False


    ## container for likelihoods and deviation at incrasing redshift
    Ps = np.zeros( [len(DMs),len(redshift_bins)] )
    devs= Ps.copy()
    ## for each redshift
    for iz, z in enumerate( redshift_bins ):
        tmp.redshift = z
        L = GetLikelihood( measure, tmp )
        if measureable:
            L.Measureable()
        Ps[:,iz], devs[:,iz] = L.Likelihoods( DMs, density=True ) ### use probability density to compare same value of DM at different redshifts. Otherwise influenced by different binning
      
    Ls = []
    for P, dev in Ps, devs:
        L = LikelihoodFunction( P=P, x=redshift_range, dev=dev ) 
        Ls.append(L)

        
    return Ls

def LikelihoodRedshiftMeasures( measures=[], datas=[], scenario=False, prior=True, renormalize=False, measurable=False ):
    """
    returns likelihood functions of redshift for observed datas of measures, 
    can be used to obtain estimate and deviation 

    Parameters
    ----------
    measure : string
        indicate which measure is probed
    data : array-like
        1D array contain extragalactic component of observed values
    scenario : dictionary
        list of models combined to one scenario
    prior : boolean
        indicate whether prior of redshift shouuld be applied
        
    """
    Ls_list = []
    
    for measure, data in zip(measures, datas):
        Ls_list.append( LikelihoodRedshiftMeasure( measure=measure, data=data, scenario=scenario, measureable=measureable ) )
        
        
        
    ## consider prior likelihood on redshift according to FRB population and telescope selection effects
    if not prior:
        pi_z = FlatPrior( measure='z', x=redshift_range )
    else:
        scenario_telescope = Scenario( population=scenario.population, telescope=scenario.telescope )
        pi_z = GetLikelihood( measure='z', scenario=scenario_telescope )
        
    Ls = []
    for P, dev in Ps, devs:
        L = LikelihoodFunction( P=P, x=redshift_range, dev=dev ) 
        L.P *= pi_z.Probability()
        L.dev = np.sqrt( L.dev**2 + pi_z.dev**2 )
        L.typ = 'posterior'
        L.Renormalize()

    Ls_result = []
    for Ls in zip( *Ls_list ):
        L = Ls[0]
        L.P = np.prod([L.P for L in Ls], axis=0) * pi_z.Probability()**prior
        L.dev = np.sqrt(np.sum([ L.dev**2 for L in Ls ], axis=0) + prior*pi_z.dev**2 )
        if prior:
            L.typ = 'posterior'
        if renormalize:
            L.Renormalize()
        Ls_result.append( L )
    return Ls_result


def LikelihoodCombined( DMs=[], RMs=[], zs=None, taus=None, scenario=False, prior=1., measureable=True, dev=False, progress_bar=False ):
    """                                                                                                                                                                                                                                                                                                            
    compute the likelihood of tuples of DM, RM (and tau) in a LoS scenario

    Parameters
    ----------
    DMs, RMs: 1D array-like of identical size,
        contain extragalactic component of observed values
    zs : array-like, len(DMs), optional
        contain redshifts of localized sources ( <= 0 for unlocalized)
    taus : array-like, len(DMs), optional
        contain temporal smearing
    scenario: Scenario-object,
        identifier for scenario
    measureable : boolean
        if True, cut the likelihood function of RM below RM_min, which cannot be observed by terrestial telescopes due to foregrounds from Galaxy and the ionosphere
    dev : boolean
        if True, also return deviation of combined likelihood, propagated from deviation of individual likelihood functions
    """

    ## container for result
    result = np.zeros_like( DMs )
    result_dev = result.copy()
    if zs is None:
        zs = np.zeros_like( DMs )
        
    ## identify localized FRBs
    localized, = np.where(zs > 0)

    ## prepare measured data
    measures = ['DM','RM']
    datas = [DMs, RMs]
    if not taus is None:
        measures.append('tau')
        datas.append(taus)
        
    ## compute combined likelihood of measures to be observed in given scenario, accounting for redshift prior
    Ps = LikelihoodRedshiftMeasures( measures=measures, datas=datas, scenario=scnenario, prior=True )  ### returns likelihood for different redshifts, need to be summed for marginal likelihood

    ### compute redshift integrated marginal likelihood for observation of measures, can be compared to other scenarios 
    Ps = [ L.Norm() for L in Ls ] 
    devs = np.sqrt( np.sum( [L.Probability()**2 * L_dev**2 for L in Ls], axis=0) )
    
    
    
    ## for localized events, instead use likelihood of DM and RM at host redshift
    tmp = scenario.copy()
    tmp.population=False
    tmp.telescope=False

    
    for loc in localized:
        tmp.redshift = zs[loc]
        L_DM = GetLikelihood( measure='DM', scenario=tmp )
        P_DM, dev_DM = L_DM.Likelihoods( measurements=[DMs[loc]], density=True )


        L_RM = GetLikelihood( measure='RM', scenario=tmp )
        if measurable: L_RM.Measureable( )
        
        P_RM, dev_RM = L_RM.Likelihoods( measurements=[RMs[loc]], density=True )
    
        result[loc] = P_DM*P_RM
        result_dev[loc] = np.sqrt( dev_DM**2 + dev_RM**2 )

    if dev:
        return Ps, dev
    return Ps
    


### !!! depreceated, remove
def LikelihoodCombined_old( DMs=[], RMs=[], zs=None, taus=None, scenario={}, prior=1., population='flat', telescope='None', measureable=True, force=False, dev=False, progress_bar=False ):
    """
    compute the likelihood of tuples of DM, RM (and tau) in a LoS scenario

    Parameters
    ----------
    DMs, RMs: 1D array-like of identical size,
        contain extragalactic component of observed values
    zs : array-like, len(DMs), optional
        contain redshifts of localized sources ( <= 0 for unlocalized)
    taus : array-like, len(DMs), optional
        contain temporal smearing
    scenario: dictionary,
        models combined to one scenario
    population: string,
        assumed cosmic population of FRBs
    telescope: string,
        instrument to observe DMs, RMs and taus
    measureable : boolean
        if True, cut the likelihood function of RM below RM_min, which cannot be observed by terrestial telescopes due to foregrounds from Galaxy and the ionosphere
    dev : boolean
        if True, also return deviation of combined likelihood, propagated from deviation of individual likelihood functions
    """

    result = np.zeros( len(DMs) )
    result_dev = result.copy()
    if zs is None:
        zs = np.zeros(len(DMs))
    localized, = np.where(zs > 0)
    
    ## estimate likelihood of source redshift based on DM and tau
#    P_redshifts_DMs, redshift_range = LikelihoodRedshift( DMs=DMs, scenario=scenario, taus=taus, population=population, telescope=telescope )
    P_redshifts_DMs, redshift_range, redshift_devs = LikelihoodRedshift( DMs=DMs, scenario=scenario, taus=taus, population=population, telescope=telescope, dev=True ) ## force=force )

#    print( 'redshift_devs', redshift_devs.mean() )

    ## for each possible source redshift
#    for redshift, P_redshift, dredshift, redshift_dev in zip( redshift_bins, P_redshifts_DMs.transpose(), np.diff(redshift_range), redshift_devs.transpose() ):
    P_redshifts_DMs = P_redshifts_DMs.transpose()
    dredshifts = np.diff(redshift_range)
    redshift_devs = redshift_devs.transpose()
    ran = trange( len(redshift_bins), desc='LikelihoodCombined' ) if progress_bar else range( len(redshift_bins) )
    for iz in ran:
        redshift, P_redshift, dredshift, redshift_dev = redshift_bins[iz], P_redshifts_DMs[iz], dredshifts[iz], redshift_devs[iz]

        ## estimate likelihood of scenario based on RM, using the redshift likelihood as a prior
        ##  sum results of all possible redshifts
        P, x, P_dev = GetLikelihood_Full( redshift=redshift, measure='RM', force=force, dev=True, **scenario )
        if measureable:
            P, x, P_dev = LikelihoodMeasureable( x=x, P=P, dev=P_dev, min=RM_min )
#        P, x = LikelihoodMeasureable( min=RM_min, typ='RM', redshift=redshift, density=False, **scenario )
#        res = P_redshift*dredshift * Likelihoods( measurements=RMs, P=P, x=x )
#        print( res)
#        result += res

        likelihoods, deviations = Likelihoods( measurements=RMs, P=P, x=x, dev=P_dev, density=True ) ### consider probability density instead of probabibilty, which cannot be compared between different scenarios due to dependence on binning
        add = P_redshift*dredshift * likelihoods
        result += add
        result_dev += add**2 * ( redshift_dev**2 + deviations**2 )
#        result += P_redshift*dredshift * Likelihoods( measurements=RMs, P=P, x=x )

#        print( 'deviations', deviations.min(), deviations.max(), deviations.mean(), P_dev.mean() )

    ## for localized events, instead use likelihood of DM and RM at host redshift
    for loc in localized:
        P, x, P_dev = GetLikelihood_Full( redshift=zs[loc], measure='DM', force=force, dev=True, **scenario )
        [result[loc]], [result_dev[loc]] = Likelihoods( measurements=[DMs[loc]], P=P, x=x, dev=P_dev, density=True )
        

        P, x, P_dev = GetLikelihood_Full( redshift=zs[loc], measure='RM', force=force, dev=True, **scenario )
        P, x, P_dev = LikelihoodMeasureable( x=x, P=P, dev=P_dev, min=measure_range['RM'][0] )
        likelihoods, deviations = Likelihoods( measurements=[RMs[loc]], P=P, x=x, dev=P_dev, density=True )
        result[loc] *= likelihoods[0]
        result_dev[loc] = result[loc]**2 * ( result_dev[loc]**2 + deviations[0]**2 )
#        result[loc] *= Likelihoods( measurements=[RMs[loc]], P=P, x=x )[0]
    result_dev = np.sqrt( result_dev ) / result 
    result *= prior
    if dev:
        return result, result_dev
    return result



def BayesFactorCombined( DMs=[], RMs=[], zs=None, scenario1={}, scenario2={}, taus=None, population='flat', telescope='None', which_NaN=False, L0=None, dev0=None, dev=False ):
    """
    for set of observed tuples of DM, RM (and tau), compute total Bayes factor that quantifies corroboration towards scenario1 above scenario2 
    first computes the Bayes factor = ratio of likelihoods for each tuple, then computes the product of all bayes factors

    Parameters
    ----------

    DMs, RMs, taus: 1D array-like of identical size,
        contain extragalactic component of observed values
    scenario1/2: Scenario-objects,
        models combined to one scenario
    population: string,
        assumed cosmic population of FRBs
    telescope: string,
        instrument to observe DMs, RMs and taus
    force : boolean,
        force new computation of convolved likelihood for scenario1. Needed after changes in the model likelihood
    force_full : boolean,
        force new computation of convolved likelihood for scenario2. Needed after changes in the model likelihood, but only once per scenario. Since scenario2 should be the same in all calls, only needed in first call
    L0 : (optional) array-like, shape(DMs), 
        provide results for scenario2 in order to accelerate computation for several scenarios 
    dev : boolean
        if True, also return deviation of combined Bayes factor, propagated from deviation of individual likelihood functions

    Return
    ------
    bayes : float
        total bayes factor
    dev : float, optional if dev==True
        log10 deviation of bayes factor
    """
    L1, dev1 = LikelihoodCombined( DMs=DMs, RMs=RMs, zs=zs, scenario=scenario1, taus=taus, dev=True )
    L2, dev2 = LikelihoodCombined( DMs=DMs, RMs=RMs, zs=zs, scenario=scenario2, taus=taus, dev=True ) if L0 is None else (L0, dev0)
    res = BayesFactor( P1=L1, P2=L2, dev1=dev1, dev2=dev2 )
    return res[:1+dev]
    '''
    ratio =  L1/L2
    NaN = np.isnan(ratio) + np.isinf(ratio)
    if np.any(NaN):
        print( "%i of %i returned NaN. Ignore in final result" %( np.sum(NaN), len(DMs) ) )
        if which_NaN:
            ix, = np.where( NaN )
            print(ix)
        return np.prod( ratio[ ~NaN ] )
    return np.prod(ratio)
#    return np.prod( LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario1, taus=taus ) / LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario2, taus=taus ) )
    '''


############################################################################
######################## READ LIKELIHOODS FROM FILE ########################
############################################################################


def GetLikelihood( measure='', scenario=False, force=False ):
    """
    return likelihood or measure in scenario
    likelihood will be read from corresponding likelihood_file
    Full and Telescope 
    
    Parameter
    ---------
    measure : string
        measure for which to return likelihood. DM, RM, SM or tau
    scenario : Scenario-object
        identifier for the scenario
    force : boolen
        if True: force new computation of likelihood and overwrite existing results in file
        only applies to Full and Telescope likelihoods

        
    Return
    ------
    L : LikelihoodFunction-object
    
    """
    file = scenario.File()
    key = scenario.Key( measure=measure )
    region = scenario.Region()
    print(region, key)
    try:
        if force and region in ['Full','Telescope']:
            gnarl  ### this causes failure and jumps to computation of likelihood            
        ## first try to read from file
        with h5.File( file, 'r') as f:
#            print('try to read')
            P, x = [f[key+'/'+axis][()] for axis in ['P', 'x'] ]
            dev = f[key+'/dev'][()] if key+'/dev' in f else []
#            print('reading works')
        L = LikelihoodFunction(  x=x, P=P, dev=dev, measure=measure, typ='prior' if region == 'redshift' else 'likelihood', scenario=scenario )
        print( L.N_sample())
        if not dev:
            L.ShotNoise( L.N_sample() )
                
        ## check for post-processing modifiers
        if region == 'IGM' and measure == 'tau' and scenario.IGM_outer_scale:
            L.Shift( (scenario.IGM_outer_scale/1000)**(-2./3) ) ## outer scale in kpc, original was computed assuming 1 Mpc
        if region == 'Inter' and scenario.N_inter:
            L.Renormalize( scenario.N_inter )
        return L
    
    except:  
        pass
    try:
        ## check for redshift independent models and modify their results, derived assuming z=0, accordingly, by applying correspoding scale factor
        if scenario.redshift == 0.0 or  region in ['Full','Telescope']:
            gnarl  ### this causes failure and jumps to computation of likelihood
        tmp = scenario.copy()
        tmp.redshift = 0.0
        L = GetLikelihood( measure, tmp )
        L.Shift( scenario.scale_factor**scale_factor_exponent[measure] )
        L.scenario.redshift = scenario.redshift
        return L
    except:
        pass

    ## compute derived likelihood and write to file
    if region == 'Telescope':
        print( "cannot find thus have to compute {}: {}".format(scenario.regions, key) )
        return ComputeTelescopeLikelihood( measure=measure, scenario=scenario )
    elif region == 'Full':
        print( "cannot find thus have to compute {}: {}".format(scenario.regions, key) )
        return ComputeFullLikelihood( measure=measure, scenario=scenario )
    ### !!! for now, assume negligible contribution of IGM for z < 0.05, use x of z=0.1 !!! 
    elif region == 'IGM' and scenario.redshift < 0.05:
        print( "due to too low redshift cannot find {}: {}. Assume contribution of IGM = 0".format(scenario.regions, key) )
        tmp = scenario.copy()
        tmp.redshift = 0.1
        L = GetLikelihood( measure, tmp )
        L.P[:] = 0
        L.scenario.redshift = scenario.redshift
        return L
    else:
        exit( "{} cannot be found in {}. \n either provide a correct scenario-key or the LikelihoodFunction for this model".format(key, file) )


### merge all this into a single simple read function  DONE
### also merge Keys in convenience to single simple function  DONE

### !!! rest of this section is depreceated, remove
def GetLikelihood_IGM( redshift=0., model='primordial', typ='far', nside=2**2, measure='DM', absolute=False, L0=1000. ):
    """ 
    read likelihood function of contribution of IGM model to measure for LoS to redshift from likelihood_file_IGM

    Parameters
    ----------
    nside : integer
        number of pixels of the healpix tesselation of the sky used to determine LoS in the constrained volume.
        for cosmological distance, redshift >= 0.1, use nside = 4
    absolue : boolean
        if True : return logarithmic likelihood of absolute value
        if False : return likelihood of value if negative values are allowed
    L0 : float
        outer scale of turbulence in kpc assumed for IGM, default: 1000 kpc (Ryu et al.2008)
        affects tau
    typ : str, 'far' or 'near', depreciated
        indicates whether to use LoS in constrained volume ('near') or cosmological LoS ('far')
    
    """
    if redshift < 0.1:
        typ='near'
        ## !!! for now, assume negligible contribution of IGM for z < 0.05, use x of z=0.1 !!!
        P, x = GetLikelihood_IGM( redshift=0.1, model=model, typ='far', nside=nside, measure=measure, absolute=absolute )
        return np.zeros(len(P)), x
        
    with h5.File( likelihood_file_IGM, 'r' ) as f:
#        print( [KeyIGM( redshift=redshift, model=model, typ=typ, nside=nside, measure='|%s|' % measure if absolute else measure, axis=axis ) for axis in ['P','x']] )
        L = [f[ KeyIGM( redshift=redshift, model=model, typ='far' if redshift >= 0.1 else 'near', nside=nside, measure='|%s|' % measure if absolute else measure, axis=axis ) ][()] for axis in ['P','x']]

    if measure=='tau' and  L0 != 1000:
        L = LikelihoodShift( *L, shift=(L0/1000)**(-2./3) ) 
    return L



def GetLikelihood_Redshift( population='SMD', telescope='None', dev=False ):
    """ read likelihood function of host redshift for population observed by telescope from likelihood_file_reshift (dev is True: also return deviation) """
    with h5.File( likelihood_file_redshift, 'r' ) as f:
        res =  [ f[ KeyRedshift( population=population, telescope=telescope, axis=axis ) ][()] for axis in ['P', 'x'] ]
    if dev:
        res.append( LikelihoodDeviation(  P=res[0], x=res[1], N=N_population[population][telescope] ) )
    return res

def GetLikelihood_HostShift( redshift=0., model='JF12', measure='DM' ):
    """ 
    read likelihood function of contribution of host model to measure for FRBs at redshift from likelihood_file_galaxy
    used for individual galaxy models, computed assuming redshift=0. likelihood function is shifted to redshift by LikelihoodShift
    """
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
#        print([ KeyHost( model=model, measure=measure, axis=axis, redshift=0.0 ) for axis in ['P', 'x'] ] )
#        return [ f[ KeyHost( model=model, measure=measure, axis=axis, redshift=0.0 ) ][()] * (1+redshift)**scale_factor_exponent[measure] for axis in ['P', 'x'] ]
        P, x = [ f[ KeyHost( model=model, measure=measure, axis=axis, redshift=0.0 ) ][()] for axis in ['P', 'x'] ]
        return LikelihoodShift( x=x, P=P, shift=(1+redshift)**-scale_factor_exponent[measure] )

def GetLikelihood_Host( redshift=0., model='Rodrigues18', measure='DM' ):
    """ 
    read likelihood function of contribution of host model to measure for FRBs at redshift from likelihood_file_galaxy
    used for galaxy models computed for individual redshift
    """
    try:
        with h5.File( likelihood_file_galaxy, 'r' ) as f:
            res = [ f[ KeyHost( model=model, redshift=redshift, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]
    except:
        print( "Host: {} shifted to z={}".format( model, redshift ) )
        res = GetLikelihood_HostShift( redshift, model,  measure )
    return res


def GetLikelihood_Inter( redshift=0., model='Rodrigues18', measure='DM', N_inter=False ):
    """ read likelihood function of contribution of intervening galaxy model to measure for LoS to redshift from likelihood_file_galaxy. Is renormalized to intersection probability if N_inter is True """
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
        P, x = [ f[ KeyInter( redshift=redshift, model=model, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]
    if N_inter:
        P *= NInter( redshift=redshift, model=model )
    return P, x

def GetLikelihood_inter( redshift=0., model='Rodrigues18', measure='DM' ):
    """ read likelihood function of contribution of intervening galaxy model to measure for LoS to redshift from likelihood_file_galaxy """
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
        return [ f[ Keyinter( redshift=redshift, model=model, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]

def GetLikelihood_Local( redshift=0., model='Piro18/uniform', measure='DM' ):
    """ read likelihood function of contribution of local environment model to measure for FRB at redshift from likelihood_file_local """
    with h5.File( likelihood_file_local, 'r' ) as f:
#        return [ f[ KeyLocal( model=model, measure=measure, axis=axis ) ][()] * (1+redshift)**scale_factor_exponent[measure] for axis in ['P', 'x'] ]
        P, x = [ f[ KeyLocal( model=model, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]
        return LikelihoodShift( x=x, P=P, shift=(1+redshift)**-scale_factor_exponent[measure] )

def GetLikelihood_MilkyWay( model='JF12', measure='DM' ):
    """ read likelihood function of contribution of Milky Way model to measure from likelihood_file_galaxy """
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
        return [ f[ KeyMilkyWay( model=model, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]


get_likelihood = {
    'IGM'        : GetLikelihood_IGM,
    'Inter'      : GetLikelihood_Inter,
    'Host'       : GetLikelihood_Host,
    'Local'      : GetLikelihood_Local,
    'MilkyWay'   : GetLikelihood_MilkyWay,  
    'MW'         : GetLikelihood_MilkyWay  
}

def GetLikelihood_old( region='IGM', model='primordial', density=True, dev=False, N_inter=False, L0=1000., smooth=True, **kwargs ):
    """ 
    read likelihood function of any individual model of region written to file
    
    Parameter
    ---------
    density : boolean
        if True: return probability density function ( 1 = sum( P * diff(x) ) )
        else: return proability function ( 1 = sum(P) )
    smooth : boolean
        if True, return smoothed P ( LikelihoodSmooth )
    L0 : float
        outer scale of turbulence in kpc assumed for IGM, default: 1000 kpc (Ryu et al.2008)
        affects tau_IGM
    **kwargs for the GetLikelihood_* function of individual regions

    Returns
    -------
    P, x(, dev) :  array-like
        likelihood P (N-array) and range x (N+1-array), renormalized such that 1 = sum( P * diff(x) )
        optional: relative deviation dev (N-array) of likelihood P

    Examples
    --------

    >>> P, x, dev =GetLikelihood_Telescope( telescope='CHIME', population='coV', measure='RM', dev=True, **{ 'IGM':['primordial'], 'host':['Rodrigues18'], 'local':['Piro18/wind'] } )
    >>> plt.errorbar( x[1:] - np.diff(x)/2, P, yerr=P*dev )

    """
    ## care for kwargs only used in one GetLikelihood_* procedure
    if region == 'IGM':
        if kwargs['measure'] == 'RM':
            kwargs['absolute'] = True
        elif kwargs['measure'] == 'tau':
            kwargs['L0'] = L0
    elif N_inter and region == 'Inter':
        kwargs['N_inter'] = True
    try:
        P, x = get_likelihood[region]( model=model, **kwargs )
    except:
        sys.exit( ("model %s in region %s is not available" % ( model, region ), "kwargs", kwargs ) )
    if not density:
        P *= np.diff(x)
    if smooth:
        P, x = LikelihoodSmooth( P=P, x=x )
    res = [P, x]
    if dev:
        res.append( LikelihoodDeviation( P=P, x=x, N=N_sample[region]  ) )
    return res
    

def GetLikelihood_Full( redshift=0.1, measure='DM', force=False, dev=False, **scenario ):
    """ 
    read likelihood function of measure for FRBs at redsift in full LoS scenario

    Parameter
    ---------
    force : boolean
        if False: try to read full likelihood function computed already
        if True: force new computation of full likelihood from individual models and write to likelihood_file_Full
    dev : boolean
        if True: also return deviation of full likelihood, according to propagation of errors of individual likelihoods
    N_inter : boolean  (in **scenario)
        if False: LoS should definitely entail an intervening galaxy  (P_Inter renormalized to 1)
        if True: it is unknown whether galaxies intersect the LoS or not (P_Inter renormalized to NInter(redshift) )
    scenario : dictionary
        contains 'region':[models] considered for the full LoS
        multiple models for same 'region' are summed, multiple regions are convolved

    Returns
    -------
    P, x(, dev) :  array-like
        likelihood P (N-array) and range x (N+1-array), renormalized such that 1 = sum( P * diff(x) )
        optional: relative deviation dev (N-array) of likelihood P

    Examples
    --------

    >>> P, x, dev = GetLikelihood_Full( redshift=1.0, measure='RM', dev=True, **{ 'IGM':['primordial'], 'host':['Rodrigues18'], 'local':['Piro18/wind'] } )
    >>> plt.errorbar( x[1:] - np.diff(x)/2, P, yerr=P*dev )
    
    """

    ## if only one model in scenario, return likelihood of that model
    if len(scenario) == 1:
        region, model = scenario.copy().popitem()
#        print('only %s' % model[0], end=' ' )
        return GetLikelihood( region=region, model=model[0], redshift=redshift, measure=measure, dev=dev )

    ## try to read from file, it may have been computed already
    if not force:
        axes = ['P','x']
        if dev:
            axes.append('dev')
        try:
            with h5.File( likelihood_file_Full, 'r' ) as f:
#                print( [ KeyFull( measure=measure, axis=axis, redshift=redshift, **scenario ) for axis in axes ] )
                return [ f[ KeyFull( measure=measure, axis=axis, redshift=redshift, **scenario ) ][()] for axis in axes ]
        except:
            print( 'cannot find P_full and have to compute', KeyFull( measure=measure, axis='P', redshift=redshift, **scenario ) )
            pass
    ## compute and write to file
    return LikelihoodFull( measure=measure, redshift=redshift, dev=dev, **scenario )

def GetLikelihood_Telescope( telescope='Parkes', population='SMD', measure='DM', force=False, dev=False, **scenario ):
    """ 
    read likelihood function of measure to be observed by telescope in case of population

    Parameter
    ---------
    force : boolean
        if False: try to read full likelihood function computed already
        if True: force new computation of full likelihood from individual models and write to likelihood_file_Full
    dev : boolean
        if True: also return deviation of full likelihood, according to propagation of errors of individual likelihoods
    N_inter : boolean (in **scenario)
        if False: LoS should definitely entail an intervening galaxy  (P_Inter renormalized to 1)
        if True: it is unknown whether galaxies intersect the LoS or not (P_Inter renormalized to NInter(redshift) )
    scenario : dictionary
        contains 'region':[models] considered for the full LoS
        multiple models for same 'region' are summed, multiple regions are convolved

    Returns
    -------
    P, x(, dev) :  array-like
        likelihood P (N-array) and range x (N+1-array), renormalized such that 1 = sum( P * diff(x) )
        optional: relative deviation dev (N-array) of likelihood P

    Examples
    --------

    >>> P, x, dev = GetLikelihood_Telescope( telescope='CHIME', population='coV', measure='RM', dev=True, **{ 'IGM':['primordial'], 'host':['Rodrigues18'], 'local':['Piro18/wind'] } )
    >>> plt.errorbar( x[1:] - np.diff(x)/2, P, yerr=P*dev )
    
    """
    L = None
    if not force:
        axes = ['P','x']
        if dev:
            axes.append('dev')
        try:
            with h5.File( likelihood_file_telescope, 'r' ) as f:
                L = [ f[ KeyTelescope( telescope=telescope, population=population, measure=measure, axis=axis, **scenario ) ][()] for axis in axes ]
#                return [ f[ KeyTelescope( telescope=telescope, population=population, measure=measure, axis=axis, **scenario ) ][()] for axis in axes ]
        except:
            print( 'cannot find P_Telescope and have to compute', KeyTelescope( telescope=telescope, population=population, measure=measure, axis='P', **scenario ) )
            pass
    if L is None:
        L = LikelihoodTelescope( population=population, telescope=telescope, measure=measure, force=force, dev=dev, **scenario )
    
    ## care for frequency dependent measures
    if measure == 'tau' and telescope == 'CHIME':  ## CHIME observes at different frequency than 1300 (assumed to estimate tau), thus has different prediction for tau propto lambda/lambda_0)^(22/5)  
        shift = ( telescope_frequency[telescope] / 1300 )**-4.4  ### this is ugly hardcoded workaround, find more beatiful solution allowing to consider frequency of individual FRBs
        L = LikelihoodShift( *L, shift=shift )
    
    return L
#    return LikelihoodTelescope( population=population, telescope=telescope, measure=measure, force=force, dev=dev, **scenario )





############################################################################
################## FAST COMPUTE LIKELIHOODS FOR SCENARIO ###################
############################################################################


def ComputeFullLikelihoods( scenario=False, models_IGMF=models_IGM[3:], N_processes=8, force=False ):
    """ 
    compute all full & telescope likelihoods for scenario, considering all redshifts, telescopes and populations. 
    For RM, also consider all models_IGMF, which do not differe in their DM, SM and tau

    Parameters
    ----------
    N_processes : integer (doesn't work)
        number of parallel processes
    force : boolean
        if True: force new computation of likelihood functions

    """
    if scenario.Region() != 'Full':
        exit( "provide a scenario for Full LoS" )

    msrs = measures[:]
    msrs.remove('RM')

    t0 = time()

    p = Pool( N_processes )

    for measure in msrs:
        f = partial( GetLikelihood, measure=measure, scenario=scenario, force=force )
#        O = p.map( f, redshift_bins )
        O=list(map( f, redshift_bins ))
#        print( len(O))
        O=0
#        p.join()

    for IGMF in models_IGMF:
        tmp = scenario.copy()
        tmp.regions['IGM'] = [IGMF]
        f = partial( GetLikelihood, measure='RM', scenario=tmp, force=force )
#        O = p.map( f, redshift_bins )
        O = list(map( f, redshift_bins ))
#        print( len(O))
        O=0
#        p.join()
    p.close()
    print( "this took %.1f minutes" % ( (time()-t0) / 60 ) )
        


def ComputeTelescopeLikelihoods( scenario={}, telescopes=telescopes, populations=populations, force=False ):
    """ 
    compute full likelihood function for LoS scenario for all redshifts and measures, as well as likelihood for all telescopes and populations
    """
    msrs = measures[:]
    msrs.remove('RM')

    t0 = time()

    for telescope in telescopes:
        for population in populations:
            tmp = scenario.copy()
            tmp.redshift= False
            tmp.population = population
            tmp.telescope = telescope
            for measure in msrs:
                GetLikelihood( measure=measure, scenario=tmp, force=force )

    print( "this took %.1f minutes" % ( (time()-t0) / 60 ) )
        


