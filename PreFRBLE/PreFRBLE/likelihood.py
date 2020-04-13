import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import trange
from PreFRBLE.convenience import *
from PreFRBLE.parameter import *
from PreFRBLE.physics import *


############################################################################
############### MATHEMATICAL LIKELIHOOD STANDARD OPERATIONS ################
############################################################################

def Likelihood( data=np.arange(1,3), bins=10, range=None, density=None, log=False, weights=None ):
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
    return h, x

Histogram = Likelihood ## old name, replace everywhere

def LikelihoodDeviation( P=[], x=[], N=1 ):
    """ compute relative deviation (Poisson noise) of likelihood function of individual model obtained from sample of N events """
    res =  ( P*np.diff(x)*N )**-0.5
    res[ np.isinf(res) + np.isnan(res)] = 0
    return res


def Likelihoods( measurements=[], P=[], x=[], dev=None, minimal_likelihood=0. ):
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

    Returns
    -------
    likelihoods: numpy array, shape( len(measurements) )
        likelihood of measurements = value of P*dx for bin, where measurement is found
    """



    likelihoods = np.zeros( len( measurements ) ) ## collector for likelihoods of measurements
    deviations = likelihoods.copy()
    Pdx = P*np.diff(x)  ## probability for obtaining measure from within bin
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
                likelihoods[i_s] = Pdx[i-1]  if i > 0 else minimal_likelihood  ## if that was the lowest bound, probability is ->zero if measurement is outside the range of P, i. e. P~0
                if dev:
                    deviations[i_s] = dev[i-1] if i > 0 else 1
                break    ## continue with the next measurement
        else:
            ## if measure is bigger than the last bin
            likelihoods[i_s] = minimal_likelihood  ## probability is zero if measurement is outside the range of P, i. e. P~0
            if dev:
                deviations[i_s] = 1
    
#    likelihoods = np.array( likelihoods )
    if dev:
        return likelihoods, deviations
    else:
        return likelihoods
        


def LikelihoodShift( x=[], P=[], shift=1. ):
    """ Shift x-values of likelihood function and renormalize accordingly: P(x|shift) = 1/shift * P(shift*x|1) """
    return P/shift, x*shift


def LikelihoodsAdd( Ps=[], xs=[], devs=[], log=True, shrink=False, weights=None, dev_weights=None, renormalize=False ):
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
        if len(devs) > 0:
            dev = devs[0]*norm
            return P, x, dev
        return P, x

    ## new function support
    l = len(Ps[0])
    if shrink:
        l = shrink
    if log:
        x = 10.**np.linspace( np.log10(np.min(xs)), np.log10(np.max(xs)), l+1 )
    else:
        x = np.linspace( np.min(xs), np.max(xs), l+1 )
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
    res = [P,x]
    if len(devs)>0:
        dev = np.sqrt(dev)/P
        dev[ np.isnan(dev) ] = 0
        res.append( dev )
    if renormalize:
        P *= renormalize/np.sum( P*np.diff(x) )
    return res

def LikelihoodShrink( P=np.array(0), x=np.array(0), dev=[], bins=100, log=True ):
    """ reduce number of bins in likelihood function, contains normalization """
    ### Actual work is done by LikelihoodsAdd, which adds up several P to new range with limited number of bins
    ### to shrink function, add P=0 with identical range
    devs = [dev,np.zeros(len(dev))] if len(dev) > 0 else []
    return LikelihoodsAdd( [P, np.zeros(len(x))], [x,x], devs=devs, shrink=bins, log=log, renormalize=np.sum( P*np.diff(x) ) )


def LikelihoodConvolve( f=np.array(0), x_f=np.array(0), g=np.array(0), x_g=np.array(0), shrink=True, log=True, absolute=False, renormalize=1 ):
    """
    compute convolution of likelihood functions f & g, i. e. their multiplied likelihood

    Parameters
    ----------
    shrink : boolean
         if True, reduce number of bins of result to standard number of bins
    log : boolean
         indicates whether x_f and x_g are log-scaled
    absolute : boolean
        indicate whether likelihood describes absolute value (possibly negative)
        if True, allow to values to cancel out by assuming same likelihood for positive and negative values

    Returns
    -------
    P, x : convolve likelihood function values and range
    """
    if absolute:
    ##   allow x-values to cancel out, assume same likelihood for + and -
#        x_min = x_g[0] + x_f[0]  ## keep minimum of resulting x for later ## this minimum is not correct, take the one below
#        x_min = np.abs(x_g[0] - x_f[0])  ## keep minimum of resulting x for later ## this is still not the minimum... solve this differently, e.g. with x[0] = ... below
        x_f = np.append( -x_f[:0:-1], np.append( 0, x_f[1:] ) )
        f = np.append( f[::-1], f )
        x_g = np.append( -x_g[:0:-1], np.append( 0, x_g[1:] ) )
        g = np.append( g[::-1], g )
    ## matrix of multiplied probabilities
    M_p = np.dot( f.reshape(len(f),1), g.reshape(1,len(g)) )
    ## matrix of combined ranges
    M_x = np.add( x_f.reshape(len(x_f),1), x_g.reshape(1,len(x_g)) )
    
    ## ranges of convolution
    x = np.unique(M_x)
    ## convolution probability
    P = np.zeros( len(x)-1 )
    ##   convolve by looping through M_p
    for i in range( len(f) ):
        for j in range( len(g) ):
    ##   for each entry, find the corresponding range in M_x
            in_ = np.where( x == M_x[i][j] )[0][0]
            out = np.where( x == M_x[i+1][j+1] )[0][0]
    ##   and add P * dx to convolved probability in that range
#            P[in_:out] += M_p[i][j]  
#            P[in_:out] += ( M_p[i][j] * (M_x[i+1][j+1] - M_x[i][j]) )
            P[in_:out] += ( M_p[i][j] * np.diff(x[in_:out+1]) )
    if absolute:
    ##   add negative probability to positive
        x = x[ x>=0] ### this makes x[0]=0, which is bad for log scale...
        x[0] = x[1]**2/x[2] ### rough, but okay... this is very close to and definitely lower than x[1] and the lowest part does not affect much the rest of the function. The important parts of the function are reproduced well
#        x = np.append( x_min, x[1+len(x)/2:] )
        P = np.sum( [ P[:int(len(P)/2)][::-1], P[int(len(P)/2):] ], axis=0 )
    ## renormalize full integral
    if renormalize:
        P *= renormalize / np.sum( P*np.diff(x) )
    if shrink:
        P, x = LikelihoodShrink( P, x, log=log )
    return P, x



def LikelihoodsConvolve( Ps=[], xs=[], devs=[], **kwargs ):
    """ 
    iteratively convolve likelihood functions
    
    Parameters
    ----------
    Ps : list
        list of values of likelihood functions
    xs : list
        list of ranges of likelihood functions
    devs : list, optional
        list of relative deviations of likelihood functions to compute relative deviation of convolved function
        !!!! ATTENTION !!!! deviation is not computed correctly if absolute=True (e. g. for RM)
    **kwargs for LikelihoodConvolve

    Returns
    -------
    P, x, (dev) : values, bin-ranges, (deviation) of renormalized convolved likelihood function 
    
    """

    ## work with probability, not pdf
    for i in range(len(Ps)):
        Ps[i] *= np.diff(xs[i])

    P, x, dev = Ps[0], xs[0], devs[0]
    for P1, x1, dev1 in zip( Ps[1:], xs[1:], devs[1:] ):
        devA  = np.sum(P1*np.diff(x1)) *  dev * P
        devB  = dev1 * P1 * np.sum( P*np.diff(x) )
        P, x = LikelihoodConvolve( P.copy(), x.copy(), P1.copy(), x1.copy(), renormalize=False, **kwargs )
        dev = np.sqrt(devA**2 + devB**2) / P
        
        ## where P=0 returns NaN. replace by 0 to not affect other data
        dev[np.isnan(dev)] = 0

    ## return renormalized pdf, not probability
    P /= np.diff(x)
    P /= np.sum(P*np.diff(x))

    return P, x, dev


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
    lo =   expect - first( zip(x, P_cum), condition= lambda x: x[1] > 0.5*(1-sigma_probability[sigma]) )[0]
    hi = - expect + first( zip(x[1:], P_cum), condition= lambda x: x[1] > 1- 0.5*(1-sigma_probability[sigma]) )[0]
    
    ## if z is clearly within one bin, hi drops negative value
    #hi = np.abs(hi)

#    x_std = np.sqrt( np.sum( P_ * ( x_ - expect)**2 ) ) ### only works for gaussian, so never

    deviation = np.array([lo,hi]).reshape([2,1])

    return expect, deviation


def WeighBayesFactor( B=1, w=1 ):
    """ Weigh the significance of Bayes factor B with weight w"""
    w_log = np.log10(w)
    return 10.**( np.log10(B) * (1+np.abs(w_log))**(1 - 2*(w_log<0) - (w_log==0) )  ) 


def BayesTotalLog( bayes, axis=None ):
    """ return log10 of total bayes factor along axis """
    return np.nansum( np.log10(bayes), axis=axis)

def BayesJackknife( bayes, axis=None ):
    """ return log10 of total bayes factor and deviation from Jackknife resampling of bayes factors of individual measurements """
    mean, dev = Jackknife( bayes, BayesTotalLog, axis=axis )
    return mean, dev

def BayesFactors( P1=0, P2=0, which_NaN=False ):
    """
    compute Bayes factors = P1/P2 between two scenarios

    Parameters
    ----------
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

def BayesFactor( P1=0, P2=0, dev1=0, dev2=0, which_NaN=False, axis=None ):
    """
    compute total Bayes factor = prod(P1/P2) between two scenarios

    Parameters
    ----------
    dev1/2 : array-like
        deviation of likelihoods P1/2, used to compute deviation of Bayes factor according to error propagation
    which_NaN : boolean
        if True, print indices of likelihoods, for which Bayes factor is NaN or infinite
    axis : integer
        if -1: return array of individual Bayes factor for each pair of P1 and P2
        if None: return total Bayes factor = product of individual Bayes factors
        else : return array of total Bayes factor computed along axis

    Returns
    -------
    bayes, dev : array-like
        Bayes factor and deviation

    """
    NaN = True
    bayes = BayesFactors( P1=P1, P2=P2, which_NaN=NaN )

    if axis == -1:
        return bayes
    
    return BayesFactorTotal( bayes, axis=axis )



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


    """
    if mode == 'simple':
        return np.prod(bayes, axis=axis)
    if mode == 'Jackknife':
        bayes, dev = BayesJackknife( bayes, axis=axis )
        return 10.**bayes, dev
    


############################################################################
#################### MATHEMATICAL LIKELIHOOD OPERATIONS ####################
############################################################################


def LikelihoodRegion( region='IGM', models=['primordial'], weights=None, **kwargs ):
    """
    return likelihood for a region. if multiple models are provided, their likelihoods are summed together 

    Parameters
    ----------
    region : string
        indicate the region along line of sight
    weights: array-like, optional
         weights to be applied to the models
    **kwargs for GetLikelihood
    """
    Ps, xs, devs = [], [], []
    for model in models:
        P, x, dev = GetLikelihood( region=region, model=model, **kwargs  )
        
        Ps.append( P )
        xs.append( x )
        devs.append( dev )
    return LikelihoodsAdd( Ps, xs, devs=devs, weights=weights )


def LikelihoodFull( measure='DM', redshift=0.1, nside_IGM=4, dev=False, **scenario ):
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

    Returns
    -------
    P, x, (dev) : likelihood function, bin ranges, (deviation)
    """

    ## collect likelihood functions for all regions along the LoS
    Ps, xs, devs = [], [], []
    for region in regions:
        model = scenario.get( region )
        if model:
            P, x, P_dev = LikelihoodRegion( region=region, models=model, measure=measure, redshift=redshift, dev=True  )
            Ps.append( P )
            xs.append( x )
            devs.append( P_dev )

#            print( 'P_dev_region', region, model, P_dev.min(), P_dev.max() )

    if len(Ps) == 0:
        sys.exit( "you must provide a reasonable scenario" )
    P, x, P_dev = LikelihoodsConvolve( Ps, xs, devs=devs, absolute= measure == 'RM' )

#    print( 'P_dev_convolve', P_dev.min(), P_dev.max(), np.array(devs).max(), measure )
    
    ## write to file
    Write2h5( likelihood_file_Full, [P,x,P_dev], [ KeyFull( measure=measure, redshift=redshift, axis=axis, **scenario ) for axis in ['P', 'x', 'dev']] )

    res = [P,x]
    if dev:
        res.append(P_dev)
    return res

'''   ### old, long and ugly version
    if len( scenario['model_MW'] ) > 0:
        P, x = LikelihoodRegion( region='MW', model=scenario['model_MW'], measure=measure  )
        Ps.append( P )
        xs.append( x )
    if len( scenario['model_IGM'] ) > 0:
<        P, x = LikelihoodRegion( 'IGM', scenario['model_IGM'], measure=measure, redshift=redshift, typ='far' if redshift >= 0.1 else 'near', nside=nside_IGM, absolute= measure == 'RM'  )
        Ps.append( P )
        xs.append( x )
    if len( scenario['model_Inter'] ) > 0:
        P, x = LikelihoodRegion( 'Inter', scenario['model_Inter'], measure=measure, redshift=redshift )
        Ps.append( P )
        xs.append( x )
    if len( scenario['model_Host'] ) > 0:
        P, x = LikelihoodRegion( 'Host', scenario['model_Host'], measure=measure, redshift=redshift, weight=scenario['weight_Host']  )
        Ps.append( P )
        xs.append( x )
    if len( scenario['model_Local'] ) > 0:
        P, x = LikelihoodRegion( 'Local', scenario['model_Local'], measure=measure, redshift=redshift  )
        Ps.append( P )
        xs.append( x )
    P, x = ConvolveProbabilities( Ps, xs, absolute= measure == 'RM', shrink=True )
    
    ## write to file
    Write2h5( likelihood_file_Full, [P,x], [ KeyFull( measure=measure, redshift=np.round(redshift,4), axis=axis, **scenario ) for axis in ['P','x']] )
    
    return P,x
'''

def LikelihoodTelescope( measure='DM', telescope='Parkes', population='SMD', nside_IGM=4, force=False, dev=False, **scenario ):
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
    for i in trange( len(redshift_bins) ):
        z = redshift_bins[i]
        P, x, dev = GetLikelihood_Full( measure=measure, redshift=z, force=force, dev=True, **scenario )
        Ps.append(P)
        xs.append(x)
        devs.append(dev)
    P, x, dev = LikelihoodsAdd( Ps, xs, devs=devs, renormalize=1., weights=Pz*np.diff(zs), dev_weights=devz )
    Write2h5( filename=likelihood_file_telescope, datas=[P,x, dev], keys=[ KeyTelescope( measure=measure, telescope=telescope, population=population, axis=axis, **scenario) for axis in ['P','x', 'dev'] ] )

    res = [P,x]
    if len(dev)>0:
        res.append(dev)
    return res



def LikelihoodMeasureable( P=[], x=[], dev=[], min=None, max=None ):
    """    returns the renormalized part of full likelihood function that can be measured by telescopes, i. e. min <= x <= max """
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


### do not load Likelhood inside function, pass it instead
def LikelihoodMeasureable_old( min=None, max=None, telescope=None, population=None, **scenario ):
    ### returns the part of full likelihood function above the accuracy of telescopes, renormalized to 1
    ###  min: minimal value considered to be measurable
    ###  kwargs: for the full likelihood
    ###  telescope: indicate survey of telescope to be predicted (requires population. If None, redshift is required)
    if telescope:
        P, x = GetLikelihood_Telescope( telescope=telescope, population=population, **scenario )
    else:
        P, x = GetLikelihood_Full( **scenario )

    if min:
        ix, = np.where( x >= min )
        x = x[ix]
        P = P[ix[:-1]] ## remember, x is range of P, i. e. size+1
        ## renormalize to 1
    if max:
        ix, = np.where( x <= max )
        x = x[ix]
        P = P[ix[:-1]] ## remember, x is range of P, i. e. size+1
        ## renormalize to 1
    P /= np.sum( P*np.diff(x) )
    return P, x


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
        Ps[:,iz], devs[:,iz] = Likelihoods( DMs, *GetLikelihood_Full( measure='DM', redshift=z, density=True, dev=True, **scenario) ) 
    
    ## improve redshift estimate with additional information from tau, which is more sensitive to high overdensities in the LoS
    ## procedure is identical, the likelihood functions are multiplied
    if taus is not None:
        Ps_ = np.zeros( [len(DMs),len(redshift_bins)] )
        devs_ = Ps_.copy()
        for iz, z in enumerate(redshift_bins):
            Ps_[:,iz], devs_[:,iz] = Likelihoods( taus, *GetLikelihood_Full( measure='tau', redshift=z, density=True, dev=True, **scenario) )  ### not all tau are measureable. However, here we compare different redshifts in the same scenario, so the amount of tau above tau_min is indeed important and does not affect the likelihood of scenarios. Instead, using LikelihoodObservable here would result in wrong estimates.
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

def LikelihoodCombined( DMs=[], RMs=[], zs=None, taus=None, scenario={}, prior=1., population='flat', telescope='None', measureable=True, force=False, dev=False ):
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

    print( 'redshift_devs', redshift_devs.mean() )

    ## for each possible source redshift
#    for redshift, P_redshift, dredshift, redshift_dev in zip( redshift_bins, P_redshifts_DMs.transpose(), np.diff(redshift_range), redshift_devs.transpose() ):
    P_redshifts_DMs = P_redshifts_DMs.transpose()
    dredshifts = np.diff(redshift_range)
    redshift_devs = redshift_devs.transpose()
    for iz in trange( len(redshift_bins) ):
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

        likelihoods, deviations = Likelihoods( measurements=RMs, P=P, x=x, dev=P_dev )
        add = P_redshift*dredshift * likelihoods
        result += add
        result_dev += add**2 * ( redshift_dev**2 + deviations**2 )
#        result += P_redshift*dredshift * Likelihoods( measurements=RMs, P=P, x=x )

        print( 'deviations', deviations.min(), deviations.max(), deviations.mean(), P_dev.mean() )

    ## for localized events, instead use likelihood of DM and RM at host redshift
    for loc in localized:
        P, x, P_dev = GetLikelihood_Full( redshift=zs[loc], measure='DM', force=force, dev=True, **scenario )
        [result[loc]], [result_dev[loc]] = Likelihoods( measurements=[DMs[loc]], P=P, x=x, dev=P_dev )
        

        P, x, P_dev = GetLikelihood_Full( redshift=zs[loc], measure='RM', force=force, dev=True, **scenario )
        P, x, P_dev = LikelihoodMeasureable( x=x, P=P, dev=P_dev, min=RM_min )
        likelihoods, deviations = Likelihoods( measurements=[RMs[loc]], P=P, x=x, dev=P_dev )
        result[loc] *= likelihoods[0]
        result_dev[loc] = result[loc]**2 * ( result_dev[loc]**2 + deviations[0]**2 )
#        result[loc] *= Likelihoods( measurements=[RMs[loc]], P=P, x=x )[0]
    result_dev = np.sqrt( result_dev ) / result 
    result *= prior
    if dev:
        return result, result_dev
    return result



def BayesFactorCombined( DMs=[], RMs=[], zs=None, scenario1={}, scenario2={}, taus=None, population='flat', telescope='None', which_NaN=False, L0=None, dev0=None, force=False, force_full=False, dev=False ):
    """
    for set of observed tuples of DM, RM (and tau), compute total Bayes factor that quantifies corroboration towards scenario1 above scenario2 
    first computes the Bayes factor = ratio of likelihoods for each tuple, then computes the product of all bayes factors

    Parameters
    ----------

    DMs, RMs, taus: 1D array-like of identical size,
        contain extragalactic component of observed values
    scenario1/2: dictionary,
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

    """
    L1, dev1 = LikelihoodCombined( DMs=DMs, RMs=RMs, zs=zs, scenario=scenario1, taus=taus, population=population, telescope=telescope, force=force, dev=True )
    L2, dev2 = LikelihoodCombined( DMs=DMs, RMs=RMs, zs=zs, scenario=scenario2, taus=taus, population=population, telescope=telescope, force=force_full, dev=True ) if L0 is None else (L0, dev0)
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

def GetLikelihood_IGM( redshift=0., model='primordial', typ='far', nside=2**2, measure='DM', absolute=False ):
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
        return [f[ KeyIGM( redshift=redshift, model=model, typ='far' if redshift >= 0.1 else 'near', nside=nside, measure='|%s|' % measure if absolute else measure, axis=axis ) ][()] for axis in ['P','x']]



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
        return LikelihoodShift( x=x, P=P, shift=(1+redshift)**scale_factor_exponent[measure] )

def GetLikelihood_Host( redshift=0., model='Rodrigues18', measure='DM' ):
    """ 
    read likelihood function of contribution of host model to measure for FRBs at redshift from likelihood_file_galaxy
    used for galaxy models computed for individual redshift
    """
    try:
        with h5.File( likelihood_file_galaxy, 'r' ) as f:
            res = [ f[ KeyHost( model=model, redshift=redshift, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]
    except:
        res = GetLikelihood_HostShift( redshift, model,  measure )
    if len(res[0]) != 100:  ### !!! change all P in file to bin=100
        res = LikelihoodShrink( *res )
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
        return LikelihoodShift( x=x, P=P, shift=(1+redshift)**scale_factor_exponent[measure] )

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

def GetLikelihood( region='IGM', model='primordial', density=True, dev=False, **kwargs ):
    """ 
    read likelihood function of any individual model of region written to file
    
    Parameter
    ---------
    density : boolean
        if True: return probability density function ( 1 = sum( P * diff(x) ) )
        else: return proability function ( 1 = sum(P) )
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
    if region == 'IGM' and kwargs['measure'] == 'RM':
        kwargs['absolute'] = True
    try:
        P, x = get_likelihood[region]( model=model, **kwargs )
    except:
        sys.exit( ("model %s in region %s is not available" % ( model, region ), "kwargs", kwargs ) )
    if not density:
        P *= np.diff(x)
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
    if not force:
        axes = ['P','x']
        if dev:
            axes.append('dev')
        try:
            with h5.File( likelihood_file_telescope, 'r' ) as f:
                return [ f[ KeyTelescope( telescope=telescope, population=population, measure=measure, axis=axis, **scenario ) ][()] for axis in axes ]
        except:
            pass
    return LikelihoodTelescope( population=population, telescope=telescope, measure=measure, force=force, dev=dev, **scenario )





############################################################################
################## FAST COMPUTE LIKELIHOODS FOR SCENARIO ###################
############################################################################



def ComputeFullLikelihood( scenario={}, models_IGMF=models_IGM[3:], N_processes=8, force=False ):
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
    msrs = measures[:]
    msrs.remove('RM')

    t0 = time()

    p = Pool( N_processes )

    for measure in msrs:
        f = partial( GetLikelihood_Full, measure=measure, force=force, **scenario )
#        O = p.map( f, redshift_bins )
        O=list(map( f, redshift_bins ))
#        print( len(O))
        O=0
#        p.join()

    for IGMF in models_IGMF:
        tmp = scenario.copy()
        tmp['IGM'] = [IGMF]
        f = partial( GetLikelihood_Full, measure='RM', force=force, **tmp )
#        O = p.map( f, redshift_bins )
        O = list(map( f, redshift_bins ))
#        print( len(O))
        O=0
#        p.join()
    p.close()
    print( "this took %.1f minutes" % ( (time()-t0) / 60 ) )
        

def ComputeTelescopeLikelihood( scenario={}, telescopes=telescopes, populations=populations, force=False ):
    """ 
    compute full likelihood function for LoS scenario for all redshifts and measures, as well as likelihood for all telescopes and populations
    """
    msrs = measures[:]
    msrs.remove('RM')

    t0 = time()

    for measure in msrs:
        for telescope in telescopes:
            for population in populations:
                GetLikelihood_Telescope( measure=measure, telescope=telescope, population=population, force=force, **scenario )

    print( "this took %.1f minutes" % ( (time()-t0) / 60 ) )
        


