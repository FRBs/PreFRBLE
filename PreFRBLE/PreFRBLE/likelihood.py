import numpy as np
from multiprocessing import Pool
from functools import partial
from PreFRBLE.convenience import *
#from PreFRBLE.parameter import *
from PreFRBLE.physics import *


## mathematical likelihood operations

def histogram( data=np.arange(1,3), bins=10, range=None, density=None, log=False ):
    ## wrapper for numpy.histogram that allows for log-scaled probability density function, used to compute likelihood function
    if log:
        if range is not None:
            range = np.log10(range)
        h, x = np.histogram( np.log10(data), bins=bins, range=range )
        x = 10.**x
        h = h.astype('float64')
        if density:
            h = h / ( np.sum( h )*np.diff(x) )
    else:
        if range is None:
            range = ( np.min(data), np.max(data) )
        h, x = np.histogram( data, bins=bins, range=range, density=density )
    return h, x



def Likelihoods( measurements=[], P=[], x=[], minimal_likelihood=0. ):
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
    minimal_likelihood : float
        value returned in case that measurement is outside x

    Returns
    -------
    likelihoods: numpy array, shape( len(measurements) )
        likelihood of measurements = value of P*dx for bin, where measurement is found
    """



    likelihoods = np.zeros( len( measurements ) ) ## collector for likelihoods of measurements
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
                break    ## continue with the next measurement
        else:
            ## if measure is bigger than the last bin
            likelihoods[i_s] = minimal_likelihood  ## probability is zero if measurement is outside the range of P, i. e. P~0
    
    likelihoods = np.array( likelihoods )
    return likelihoods


def LikelihoodsAdd( Ps=[], xs=[], log=True, shrink=False, weights=None, renormalize=False ):
    ### add together several likelihoos functions
    ###  Ps: list of likelihood functions
    ###  xs: list of bin ranges of likelihood functions
    ###  log: set to False if xs are not log-scaled
    ###  shrink=bins: force number of bins in result, otherwise use size of first likelihood function
    ###  weights: provide weights for the likelihood functions
    ### renormalize: total likelihood of the final result

    if len(Ps) == 1:
        ## if only one function is given, return the original
        P, x = Ps[0], xs[0] 
        if renormalize: ## maybe renormalized to new value
            P *= renormalize/np.sum( P*np.diff(x) )
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
        
    P = np.zeros( l )

    ## for each function
    for f, x_f, w in zip(Ps, xs, weights):
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
#                P[ib] += f[ix]  ## add if one
                P[ib] += w * f[ix]  ## add if one
            else:  ## compute average of contributing bins
    ##     get corresponding ranges
                x_ = x_f[np.append(ix,ix[-1]+1)]
    ##     restrict range to within target bin
                x_[0], x_[-1] = b0, b1
    ##     add weighed average to target likelihood
                P[ib] += w * np.sum( f[ix]*np.diff(x_) ) / (b1-b0)
    if renormalize:
        P *= renormalize/np.sum( P*np.diff(x) )
    return P, x

def LikelihoodShrink( P=np.array(0), x=np.array(0), bins=100, log=True ):
    ### reduce number of bins in likelihood function, contains normalization
    ### Actual work is done by LikelihoodsAdd, which adds up several P to new range wit limited number of bins
    ### to shrink function, add P=0 with identical range
    return LikelihoodsAdd( [P, np.zeros(len(x))], [x,x], shrink=bins, log=log, renormalize=np.sum( P*np.diff(x) ) )


def LikelihoodConvolve( f=np.array(0), x_f=np.array(0), g=np.array(0), x_g=np.array(0), shrink=True, log=True, absolute=False ):
    ### compute convolution of likelihood functions f & g, i. e. their multiplied likelihood
    ###  shrink=True: number of bins of result reduced to number of bins in f (will get big otherwise)
    ###  log: indicates whether x_f and x_g are log-scaled
    ###  absolute: if likelihood is for absolute value, allow to cancel out! assume same likelihood for + and -

    if absolute:
    ##   allow values to cancel out, assume same likelihood for + and -
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
    ##   and add probability to convolved probability in that range
            P[in_:out] += M_p[i][j]
    if absolute:
    ##   add negative probability to positive
        x = x[ x>=0] ### this makes x[0]=0, which is bad for log scale...
        x[0] = x[1]**2/x[2] ### rough, but okay... this is very close to and definitely lower than x[1] and the lowest part does not affect much the rest of the function. The important parts of te function are reproduced well
#        x = np.append( x_min, x[1+len(x)/2:] )
        P = np.sum( [ P[:int(len(P)/2)][::-1], P[int(len(P)/2):] ], axis=0 )
    ## renormalize full integral to 1
    P /= np.sum( P*np.diff(x) )
    if shrink:
        return LikelihoodShrink( P, x, bins=len(f), log=log )
    else:
        return P, x



def LikelihoodsConvolve( Ps=[], xs=[], **kwargs ):
    ### iteratively convolve likelihood functions 
    ###  kwargs for Convole Probability
    P, x = Ps[0], xs[0]
    i = 0.
    for P_, x_ in zip( Ps[1:], xs[1:] ):
        P, x = LikelihoodConvolve( P, x, P_, x_, **kwargs )
        i += 1
        P /= np.sum( P*np.diff(x) )
    return P, x


def LikelihoodRegion( region='IGM', models=['primordial'], weights=None, **kwargs ):
    ### return likelihood for region, if multiple models are provided, their likelihoods are summed together 
    ###  weights: weights to be applied to the models
    ###  kwargs: for GetLikelihood
    Ps, xs = [], []
    for model in models:
        P, x = GetLikelihood( region=region, model=model, **kwargs  )
        
        Ps.append( P )
        xs.append( x )
    return LikelihoodsAdd( Ps, xs, weights=weights )


def LikelihoodFull( measure='DM', redshift=0.1, nside_IGM=4, **scenario ):
    ### return the full likelihood function for measure in the given scenario
    ###  redshift: of the source
    ###  nside_IGM: pixelization of IGM full-sky maps

    ## collect likelihood functions for all regions along the LoS
    Ps, xs = [], []
    for region in regions:
        model = scenario.get( region )
        if model:
            P, x = LikelihoodRegion( region=region, models=model, measure=measure, redshift=redshift  )
            Ps.append( P )
            xs.append( x )
    if len(Ps) == 0:
        sys.exit( "you must provide a reasonable scenario" )
    P, x = LikelihoodsConvolve( Ps, xs, absolute= measure == 'RM' )
    
    ## write to file
    Write2h5( likelihood_file_Full, [P,x], [ KeyFull( measure=measure, redshift=redshift, axis=axis, **scenario ) for axis in ['P','x']] )
    
    return P,x

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

def LikelihoodTelescope( measure='DM', telescope='Parkes', population='SMD', nside_IGM=4, **scenario ):
    ### return the likelihood function for measure expected to be observed by telescope
    ###  nside_IGM: pixelization of IGM full-sky maps
        
    ## prior on redshift is likelihood based on FRB population and telescope selection effects 
    if population == 'flat':
        Pz = None
    else:
        Pz, zs = GetLikelihood_Redshift( population=population, telescope=telescope )
    
    ## possible solutions for all redshifts are summed, weighed by the prior
    Ps, xs = [], []
    for z in redshift_bins:
        P, x = GetLikelihood_Full( measure=measure, redshift=z, **scenario )
        Ps.append(P)
        xs.append(x)
    P, x = LikelihoodsAdd( Ps, xs, renormalize=1., weights=Pz*np.diff(zs) )
    Write2h5( filename=likelihood_file_telescope, datas=[P,x], keys=[ KeyTelescope( measure=measure, telescope=telescope, population=population, axis=axis, **scenario) for axis in ['P','x'] ] )
    return P, x



def LikelihoodMeasureable( P=[], x=[], min=None, max=None ):
    ### returns the part of full likelihood function above the accuracy of telescopes, renormalized to 1
    ###  min: minimal value considered to be measurable
    ###  max: maximal value considered to be measurable
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


def LikelihoodRedshift( DMs=[], scenario={}, taus=None, population='flat', telescope='None' ):
    ### returns likelihood functions of redshift for observed DMs (and taus)
    ### can be used to obtain estimate and deviation
    ###  DMs, SMs: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario: dictionary of models combined to one scenario
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and taus

    Ps = np.zeros( [len(DMs),len(redshift_bins)] )
    ## for each redshift
    for iz, z in enumerate( redshift_bins ):
        ## calculate the likelihood of observed DM 
        Ps[:,iz] = Likelihoods( DMs, *GetLikelihood_Full( measure='DM', redshift=z, density=True, **scenario) ) 
    
    ## improve redshift estimate with additional information from tau, which is more sensitive to high overdensities in the LoS
    ## procedure is identical, the likelihood functions are multiplied
    if taus is not None:
        Ps_ = np.zeros( [len(DMs),len(redshift_bins)] )
        for iz, z in enumerate(redshift_bins):
            Ps_[:,iz] = Likelihoods( taus, *GetLikelihood_Full( measure='tau', redshift=z, density=True, **scenario) )  ### not all tau are measureable. However, here we compare different redshifts in the same scenario, so the amount of tau above tau_min is indeed important and does not affect the likelihood of scenarios. Instead, using LikelihoodObservable here would result in wrong estimates.
        Ps *= Ps_
        Ps_= 0
    
    ## consider prior likelihood on redshift according to FRB population and telescope selection effects 
    if population == 'flat':
        pi, x = np.array([1.]), np.arange(2)
    else:
        pi, x = GetLikelihood_Redshift( population=population, telescope=telescope )
    Ps = Ps * np.resize( pi*np.diff(x), [1,len(redshift_bins)] )
                    
    ## renormalize to 1 for every DM (only if any P is not zero)
    for P in Ps:
        if np.any( P > 0):
            P /= np.sum( P*np.diff( redshift_range ) )

#    Ps = Ps / np.resize( np.sum( Ps * np.resize( np.diff( redshift_range ), [1,len(redshift_bins)] ), axis=1 ), [len(DMs),1] )

    return Ps, redshift_range

def LikelihoodCombined( DMs=[], RMs=[], taus=None, scenario={}, prior=1., population='flat', telescope='None' ):
    ### compute the likelihood of tuples of DM, RM (and tau) in a LoS scenario
    ###  DMs, RMs, taus: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario: dictionary of models combined to one scenario
    ###  prior: prior attributed to scenario
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and taus


    result = np.zeros( len(DMs) )
    
    ## estimate likelihood of source redshift based on DM and tau
    P_redshifts_DMs, redshift_range = LikelihoodRedshift( DMs=DMs, scenario=scenario, taus=taus, population=population, telescope=telescope )
    
    ## for each possible source redshift
    for redshift, P_redshift, dredshift in zip( redshift_bins, P_redshifts_DMs.transpose(), np.diff(redshift_range) ):
        ## estimate likelihood of scenario based on RM, using the redshift likelihood as a prior
        ##  sum results of all possible redshifts
        P, x = GetLikelihood_Full( measure='RM', **scenario )
        P, x = LikelihoodMeasureable( x=x, P=P, min=RM_min )
#        P, x = LikelihoodMeasureable( min=RM_min, typ='RM', redshift=redshift, density=False, **scenario )
#        res = P_redshift*dredshift * Likelihoods( measurements=RMs, P=P, x=x )
#        print( res)
#        result += res
        result += P_redshift*dredshift * Likelihoods( measurements=RMs, P=P, x=x )
 
    return result * prior



def BayesFactorCombined( DMs=[], RMs=[], scenario1={}, scenario2={}, taus=None, population='flat', telescope='None', which_NaN=False ):
    ### for set of observed tuples of DM, RM (and tau), compute total Bayes factor that quantifies corroboration towards scenario1 above scenario2 
    ### first computes the Bayes factor = ratio of likelihoods for each tuple, then computes the product of all bayes factors
    ###  DMs, RMs, taus: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario1/2: dictionary of models combined to one scenario
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and taus
    L1 = LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario1, taus=taus )
    L2 = LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario2, taus=taus )
    result =  np.prod(L1/L2)
    NaN = np.isnan(result)
    if np.any(NaN):
        print( "%i of %i returned NaN. Ignore in final result" %( np.sum(NaN), len(DMs) ) )
        if which_NaN:
            ix, = np.where( NaN )
            print(ix)
        return np.nanprod( L1/L2 )
    return result
#    return np.prod( LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario1, taus=taus ) / LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario2, taus=taus ) )


def Likelihood2Expectation( P=np.array(0), x=np.array(0), log=True,  density=True, sigma=1, std_nan=np.nan ):
    """
    computes the estimate value and deviation from likelihood function P (must be normalized to 1)


    Paraeters
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


## GetLikelihood functions

## read likelihood function from file
def GetLikelihood_IGM( redshift=0., model='primordial', typ='far', nside=2**2, measure='DM', absolute=False ):
    if redshift < 0.1:
        typ='near'
    with h5.File( likelihood_file_IGM, 'r' ) as f:
         return [f[ KeyIGM( redshift=redshift, model=model, typ=typ, nside=nside, measure='|%s|' % measure if absolute else measure, axis=axis ) ][()] for axis in ['P','x']]



def GetLikelihood_Redshift( population='SMD', telescope='None' ):
    with h5.File( likelihood_file_redshift, 'r' ) as f:
        return [ f[ KeyRedshift( population=population, telescope=telescope, axis=axis ) ][()] for axis in ['P', 'x'] ]

def GetLikelihood_Host_old( redshift=0., model='JF12', weight='uniform', measure='DM' ):
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
        return [ f[ KeyHost( model=model, weight=weight, measure=measure, axis=axis ) ][()] * (1+redshift)**scale_factor_exponent[measure] for axis in ['P', 'x'] ]

def GetLikelihood_Host( redshift=0., model='Rodrigues18/smd', measure='DM' ):
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
        return [ f[ KeyHost( model=model, redshift=redshift, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]


def GetLikelihood_Inter( redshift=0., model='Rodrigues18', measure='DM' ):
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
        return [ f[ KeyInter( redshift=redshift, model=model, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]

def GetLikelihood_Local( redshift=0., model='Piro18/uniform', measure='DM' ):
    with h5.File( likelihood_file_local, 'r' ) as f:
        return [ f[ KeyLocal( model=model, measure=measure, axis=axis ) ][()] * (1+redshift)**scale_factor_exponent[measure] for axis in ['P', 'x'] ]

def GetLikelihood_MilkyWay( model='JF12', measure='DM' ):
    with h5.File( likelihood_file_galaxy, 'r' ) as f:
        return [ f[ KeyMilkyWay( model=model, measure=measure, axis=axis ) ][()] for axis in ['P', 'x'] ]


get_likelihood = {
    'IGM'  :       GetLikelihood_IGM,
    'Inter' :      GetLikelihood_Inter,
    'Host' :       GetLikelihood_Host,
    'Local' : GetLikelihood_Local,
    'MilkyWay'   : GetLikelihood_MilkyWay,  
    'MW'         : GetLikelihood_MilkyWay  
}

def GetLikelihood( region='IGM', model='primordial', density=True, **kwargs ):
    ## wrapper to read any likelihood function written to file
    if region == 'IGM' and kwargs['measure'] == 'RM':
        kwargs['absolute'] = True
    P, x = get_likelihood[region]( model=model, **kwargs )
    if not density:
        P *= np.diff(x)
    return P, x

def GetLikelihood_Full( redshift=0.1, measure='DM', force=False, **scenario ):

    if len(scenario) == 1:
        region, model = scenario.copy().popitem()
#        print('only %s' % model[0], end=' ' )
        return GetLikelihood( region=region, model=model[0], redshift=redshift, measure=measure )
    if not force:
        try:
            with h5.File( likelihood_file_Full, 'r' ) as f:
                return [ f[ KeyFull( measure=measure, axis=axis, redshift=redshift, **scenario ) ][()] for axis in ['P', 'x'] ]
        except:
            pass
    return LikelihoodFull( measure=measure, redshift=redshift, **scenario )

def GetLikelihood_Telescope( telescope='Parkes', population='SMD', measure='DM', force=False, **scenario ):
    if not force:
        try:
            with h5.File( likelihood_file_telescope, 'r' ) as f:
                return [ f[ KeyTelescope( telescope=telescope, population=population, measure=measure, axis=axis, **scenario ) ][()] for axis in ['P', 'x'] ]
        except:
            pass
    return LikelihoodTelescope( population=population, telescope=telescope, measure=measure, **scenario )




### procedures for fast parallel computation of combined likelihood functions

def ComputeFullLikelihood( scenario={}, models_IGMF=models_IGM[3:], N_processes=8, force=False ):
    ### compute fill likelihood functions for all redshifts and measures in scenario
    ### for RM, also investigate all models_IGMF with identical DM, SM and tau as model_IGM in scenario
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
        


