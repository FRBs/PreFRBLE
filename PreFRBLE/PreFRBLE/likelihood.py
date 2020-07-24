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
            L.Smooth()
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
    L.Likelihood( np.abs( np.sum( samples, axis=0 ) ), log=Ls[0].log, bins=Ls[0].P.size )
    if smooth:
        L.Smooth()
    L.ShotNoise( N=N )
    return L


def WeighBayesFactor( bayes=1, weight=1 ):
    """ Weigh the significance of Bayes factor bayes with weight w"""
    w_log = np.log10(weight)
    return 10.**( np.log10(bayes) * (1+np.abs(w_log))**(1 - 2*(w_log<0) - (w_log==0) )  ) 


def BayesTotalLog( bayes, axis=None ):
    """ return log10 of total bayes factor along axis """
    return np.nansum( np.log10(bayes), axis=axis)

def BayesJackknife( bayes, axis=None ):
    """ return log10 of total bayes factor and deviation from Jackknife resampling of bayes factors of individual measurements """
    mean, dev = Jackknife( bayes, BayesTotalLog, axis=axis )
    return mean, dev

def BayesFactors( P1=0, P2=0, dev1=[], dev2=[], which_NaN=False ):
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
    if len(dev1) and len(dev2):
        dev = np.sqrt( dev1**2 + dev2**2 )
    NaN = np.isnan(bayes) + np.isinf(bayes)
#    dev = np.sqrt( dev1*2 + dev2**2 )
    if np.any(NaN):
        print( "%i of %i returned NaN. Ignore in final result" %( np.sum(NaN), len(bayes) ) )
        bayes[NaN] = 1
        if len(dev1) and len(dev2):
            dev[NaN] = 0
        if which_NaN:
            which_NaN = np.where( NaN )
            print(which_NaN)
    else:
        which_NaN = None
    if len(dev1) and len(dev2):
        return bayes, dev
    return bayes

def BayesFactor( P1=0, P2=0, dev1=0, dev2=0, which_NaN=False, axis=None, mode='Jackknife' ):
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
    bayes = BayesFactors( P1=P1, P2=P2, dev1=dev1, dev2=dev2, which_NaN=NaN )
    if len(dev1) and len(dev2):
        bayes, bayes_dev = bayes
    else:
        bayes_dev = []
    if axis == -1:
        return bayes
    
    return BayesFactorTotal( bayes, bayes_dev, axis=axis, mode=mode )



def BayesFactorTotal( bayes, dev=[], mode='Jackknife', axis=None ):
    """ 
    return total bayes factor using mode

    Parameter
    ---------
    bayes : array-like
        individual bayes factors 
    dev : array-like
        deviation of individual bayes factors 
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
        tot = np.prod(bayes, axis=axis)
        if len(dev):
            dev = np.sqrt( np.sum( dev**2 ) )
            return tot, dev
        return tot
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
        properties = scenario.Properties( regions=False )
        properties.update( {region:model} )
        tmp = Scenario( **properties )
        L = GetLikelihood( measure=measure, scenario=tmp )
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
    if scenario.population == 'flat':
        pi_z = FlatPrior( measure='z', x=redshift_range )
    else:
        scenario_telescope = Scenario( **scenario.Properties( regions=False ) )
        pi_z = GetLikelihood( measure='z' , scenario=scenario_telescope)

    ## possible solutions for all redshifts are summed, weighed by the prior
    Ls = []

    ## prepare scenario used for full likelihood function at increasing redshift
    tmp = Scenario( redshift=1.0, **scenario.Properties( identifier=False ) )

    ## optionally, provide progress bar
    l = len(redshift_bins)
    ran = trange( l, desc='LikelihoodTelescope {} {}'.format( telescope, population ) ) if progress_bar else range( l )
    for i in ran:
        tmp.redshift = redshift_bins[i]
        L = GetLikelihood( measure=measure, scenario=tmp, force=force )
#        L.Smooth()
        Ls.append(L)
    L = LikelihoodsAdd( *Ls, weights=pi_z.Probability(), dev_weights=pi_z.dev )
    L.scenario = scenario
    L.Write()
    return L


### !!! depreceated, remove
### instead, use  LikelihoodRedshiftMeasure( datas=[data], **kwargs)[0]
def LikelihoodRedshiftMeasure( measure='', data=0.0, scenario=False, measureable=False, prior=True, renormalize=False ):
    """
    returns likelihood functions of redshift for observed data of measure, 
    can be used to obtain estimate and deviation 

    Parameters
    ----------
    measure : string
        indicate which measure is probed
    data : float
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
    tmp = Scenario( redshift=1.0, **scenario.Properties( identifier=False ) )

    ## container for likelihoods and deviation at incrasing redshift
    P = np.zeros( [len(redshift_bins)] )
    dev= P.copy()
    ## for each redshift
    for iz, z in enumerate( redshift_bins ):
        tmp.redshift = z
        L = GetLikelihood( measure, tmp )
        if measureable:
            L.Measureable()
        P[iz], dev[iz] = L.Likelihoods( [data], density=True, deviation=True ) ### use probability density to compare same values at different redshifts or in different scenarios. Otherwise influenced by different binning
      
    L = LikelihoodFunction( measure='z', P=P, x=redshift_range, dev=dev )
    if not prior or scenario.population is 'flat':
        L.scenario = Scenario( population='flat', telescope='None', **scenario.Properties( identifier=False ) )
        if renormalize:
            L.Renormalize()
        return L

    ## obtain redshift prior
    scenario_telescope = Scenario( **scenario.Properties( parameter=False, regions=False ) )
    pi_z = GetLikelihood( measure='z', scenario=scenario_telescope )

    ## apply prior to likelihood
    L.P *= pi_z.Probability()
    L.dev = np.sqrt( L.dev**2 + pi_z.dev**2 )
    L.typ = 'posterior'
    if renormalize:
        L.Renormalize()
    L.scenario = scenario
    return L

def LikelihoodRedshiftMeasures( measure='', datas=[], scenario=False, measureable=False, prior=True, renormalize=False, smooth=True ):
    """
    returns likelihood functions of redshift for observed data of measure, 
    can be used to obtain estimate and deviation 

    Parameters
    ----------
    measure : string
        indicate which measure is probed
    data : float
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
    tmp = Scenario( redshift=1.0, **scenario.Properties( identifier=False ) )

    ## container for likelihoods and deviation at incrasing redshift
#    P = np.zeros( [len(redshift_bins)] )
#    dev= P.copy()
    Ps, devs = [], []


    ## for each redshift
    for iz, z in enumerate( redshift_bins ):
        tmp.redshift = z
        L = GetLikelihood( measure, tmp )
        if smooth:
            L.Smooth()
        if measureable:
            L.Measureable()
        P, dev = L.Likelihoods( datas, density=True, deviation=True ) ### use probability density to compare same value at different redshifts. Otherwise influenced by different binning
        Ps.append(P)
        devs.append(dev)
#        P[iz], dev[iz] = L.Likelihoods( [data], density=True, deviation=True ) ### use probability density to compare same value of DM at different redshifts. Otherwise influenced by different binning
      
    Ps = np.array(Ps).T
    devs = np.array(devs).T
    Ls = [ LikelihoodFunction( measure='z', P=P, x=redshift_range, dev=dev ) for P, dev in zip( Ps, devs ) ]

#    L = LikelihoodFunction( measure='z', P=P, x=redshift_range, dev=dev )
    if not prior or scenario.population is 'flat':
        for i in range(len(Ls)):
            Ls[i].scenario = Scenario( population='flat', telescope='None', **scenario.Properties( identifier=False ) )
            if renormalize:
                Ls[i].Renormalize()
        return Ls

    ## obtain redshift prior
    scenario_telescope = Scenario( **scenario.Properties( parameter=False, regions=False ) )
    pi_z = GetLikelihood( measure='z', scenario=scenario_telescope )

    ## apply prior to likelihood
    for i in range(len(Ls)):
        Ls[i].P *= pi_z.Probability()
        Ls[i].dev = np.sqrt( Ls[i].dev**2 + pi_z.dev**2 )
        Ls[i].typ = 'posterior'
        if renormalize:
            Ls[i].Renormalize()
        Ls[i].scenario = scenario
    return Ls
        
def RedshiftEstimate( DM=0.0, scenario=False, deviation=False, sigma=1 ):
    """ estimate host redshift (and sigma deviation) from extragalactic DM assuming scenario"""
    L = LikelihoodRedshiftMeasure( 'DM', DM, scenario, prior=True, renormalize=True )
    res = L.Expectation( sigma=sigma )
    return res[:1+deviation]    



def LikelihoodCombined( measures=[], datas=[], zs=[], scenario=False, measureable=True, deviation=False, progress_bar=False ):
    """                                                                                                                                                                                                                                                                                                            
    compute the likelihood of tuples of DM, RM (and tau) in a LoS scenario

    Parameters
    ----------
    measures : array-like
        list of measures to be combined
    datas : array-like, ( len(measures), N_data )
        list of data arrays observed for measures
    zs : array-like, len(DMs), optional
        contain redshifts of localized sources ( <= 0 for unlocalized)
    scenario: Scenario-object,
        identifier for scenario
    measureable : boolean
        if True, cut the likelihood function of RM below RM_min, which cannot be observed by terrestial telescopes due to foregrounds from Galaxy and the ionosphere
    deviation : boolean
        if True, also return deviation of combined likelihood, propagated from deviation of individual likelihood functions
    """

    ## identify localized FRBs
    localized = np.where(zs > 0)[0] if len(zs) else []

    ## compute combined likelihood of measures to be observed in given scenario from source at different redshifts, accounting for redshift prior
    Ls = []
    for measure, data in zip(measures, datas):
        Ls.append( LikelihoodRedshiftMeasures( measure, data, scenario=scenario, prior=True, renormalize=False ) )


    ## combine likelihood of measures and compute marginal likelihood, i. e. integrate all possible redshift, which can be compared to other scenarios
    Ps, devs = [], [] ### container for combined likelihood
    for L in zip(*Ls):
        L_combined = L[0]
        L_combined.P = np.prod( [LL.P for LL in L], axis=0 )
        L_combined.dev = np.sqrt( np.sum( [LL.Probability()**2 * LL.dev**2 for LL in L], axis=0 ) )
        Ps.append( L_combined.Norm() ) ## collect marginal likelihood
        devs.append( L_combined.CumulativeDeviation()[-1] ) 

    Ls = 0 ## free memory

    Ps, devs = np.array( Ps ), np.array( devs )
    
    ## for localized events, instead use likelihood of DM and RM at host redshift
    tmp = Scenario( redshift=1.0, **scenario.Properties( identifier=False ) )
    
    for loc in localized:
        tmp.redshift = zs[loc]
        L_DM = GetLikelihood( measure='DM', scenario=tmp )
        P_DM, dev_DM = L_DM.Likelihoods( measurements=[DMs[loc]], density=True )


        L_RM = GetLikelihood( measure='RM', scenario=tmp )
        if measurable: L_RM.Measureable( )
        
        P_RM, dev_RM = L_RM.Likelihoods( measurements=[RMs[loc]], density=True )
    
        Ps[loc] = P_DM*P_RM
        devs[loc] = np.sqrt( dev_DM**2 + dev_RM**2 )

    if deviation:
        return Ps, devs
    return Ps
    




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


def GetLikelihood( measure='', scenario=False, force=False, echo=False ):
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
    if echo:
        print(region, key)
    try:
#    if True:
        if force and region in ['Full','Telescope']:
            gnarl  ### this causes failure and jumps to computation of likelihood            
        ## first try to read from file
        with h5.File( file, 'r') as f:
            if echo:
                print('try to read')
            P, x = [f[key+'/'+axis][()] for axis in ['P', 'x'] ]
            dev = f[key+'/dev'][()] if key+'/dev' in f else []
            if echo:
                print('reading works')
        L = LikelihoodFunction(  x=x, P=P, dev=dev, measure=measure, typ='prior' if region == 'redshift' else 'likelihood', scenario=scenario )
        if not len(dev):
            if echo:
                print("compute ShotNoise")
            L.ShotNoise( L.N_sample() )
                
        if echo:
            print("post-processing modifiers")
        ## check for post-processing modifiers
        if region == 'IGM' and scenario.f_IGM:
            L.Shift( scenario.f_IGM )
        if region == 'IGM' and measure == 'tau' and scenario.IGM_outer_scale:
            L.Shift( (scenario.IGM_outer_scale/1000)**(-2./3) ) ## outer scale in kpc, original was computed assuming 1 Mpc
        if region == 'Inter' and scenario.N_inter:
            L.Renormalize( scenario.N_inter )

        if scenario.telescope == 'CHIME'  and measure == 'tau':
            L.Shift( ( telescope_frequency[scenario.telescope]/1300)**(-4.4) )
        if echo:
            print('all clear')
        return L
    
    except:  
        pass
    try:
        ## check for redshift independent models and modify their results, derived assuming z=0, accordingly, by applying correspoding scale factor
        if scenario.redshift == 0.0 or  region in ['Full','Telescope']:
            gnarl  ### this causes failure and jumps to computation of likelihood
        tmp = Scenario( redshift=0.0, **scenario.Properties( identifier=False ) )
        L = GetLikelihood( measure, tmp )
        L.Shift( scenario.scale_factor**scale_factor_exponent[measure] )
        L.scenario.redshift = scenario.redshift
        return L
    except:
        pass

    ## compute derived likelihood and write to file
    if region == 'Telescope':
        if not force:
            print( "cannot find thus have to compute {}: {} \nrerun for frequency dependent measures and telescopes that do not probe 1300 MHz".format(scenario.regions, key) )
        return ComputeTelescopeLikelihood( measure=measure, scenario=scenario )
    elif region == 'Full':
        if not force:
            print( "cannot find thus have to compute {}: {}".format(scenario.regions, key) )
        return ComputeFullLikelihood( measure=measure, scenario=scenario )
    ### !!! for now, assume negligible contribution of IGM for z < 0.05, use x of z=0.1 !!! 
    elif region == 'IGM' and scenario.redshift < 0.05:
        print( "due to too low redshift cannot find {}: {}. Assume contribution of IGM = 0".format(scenario.regions, key) )
        tmp = Scenario( redshift=0.1, **scenario.Properties( identifier=False ) )
        L = GetLikelihood( measure, tmp )
        L.P[:] = 0
        L.scenario.redshift = scenario.redshift
        return L
    else:
        exit( "{} cannot be found in {}. \n either provide a correct scenario-key or the LikelihoodFunction for this model".format(key, file) )



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
            tmp = Scenario( population=population, telescope=telescope, **scenario.Properties( identifier=False ) )
            for measure in msrs:
                GetLikelihood( measure=measure, scenario=tmp, force=force )

    print( "this took %.1f minutes" % ( (time()-t0) / 60 ) )
        


