import sys, h5py as h5, numpy as np, matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from matplotlib import colors, cm


RM_min = 1 # rad m^-2  ## minimal RM measureable by telescopes

units = {
    'DM'       :r"pc cm$^{-3}$",
    'RM'       :r"rad m$^{-2}$",
    'SM'       :r"kpc m$^{-20/3}$",
    'z'        :r"z",
    'redshift' :r"1+z",
}


## main working folder
root = '/work/stuf315/PreFRBLE/results/'
root = '/hummel/PreFRBLE/'
#root = '/media/hqi/6A57-6B65/PreFRBLE/results/'

root_likelihood  = root + 'likelihood/'
root_results = root + 'results/'

likelihood_file = root_likelihood+'observables_likelihood.h5'
sky_file = root_results+'observables_maps_galaxy.h5'

likelihood_file_progenitor = root_likelihood+'observables_likelihood_progenitor.h5'
likelihood_file_galaxy = root_likelihood+'observables_likelihood_galaxy.h5'
likelihood_file_IGM = root_likelihood+'observables_likelihood_IGM.h5'
likelihood_file_redshift = root_likelihood+'redshift_likelihood.h5'

likelihood_file_Full = root_likelihood+'observables_likelihood_Full.h5'
#likelihood_file_Full = root_likelihood+'DMRMprobability_Full.h5'



## data keys inside likelihood files
def KeyProgenitor( model='Piro18/wind', measure='DM', axis='P' ):
    return '/'.join( [ model, measure, axis ] )

def KeyMilkyWay( model='JF12', measure='DM', axis='P'  ):
    return '/'.join( [ 'MilkyWay', model, measure, axis ] )

def KeyHost( redshift=0.0, model='Rodrigues18', weight='smd', measure='DM', axis='P' ):
    return '/'.join( [ 'Host', model, weight, '%.4f' % redshift, measure, axis ] )

def KeyInter( redshift=0.0, model='Rodrigues18', measure='DM', axis='P' ):
    return '/'.join( [ 'Intervening', model, '%.4f' % redshift, measure, axis ] )

def KeyIGM( redshift=0.1, model='primordial', typ='far', nside=2**2, measure='DM', axis='P' ):  ## nside=2**6
    return '/'.join( [ model, typ, str(nside), measure, '%.4f' % redshift, axis ] )

def KeyRedshift( population='flat', telescope='none', axis='P' ):
    return '/'.join( [ population, telescope, axis] )

#def KeyFull( measure='DM', axis='P', redshift=0.1, model_MW=['JF12'], model_IGM=['primordial'], model_Host=['Heesen11/IC10'], weight_Host='StarDensity_MW', model_Progenitor=['Piro18/uniform_JF12'] ):
def KeyFull( measure='DM', axis='P', redshift=0.1, **scenario ):
    models = np.append( scenario['model_MW'], scenario['model_IGM'] )
    models = np.append( models, scenario['model_Host'] )
    models = np.append( models, scenario['weight_Host'] )
    models = np.append( models, scenario['model_Progenitor'] )
    models = np.append( models, [redshift, measure,axis] )
    return '/'.join( models )


## wrapper to write hdf5 files consistently
def Write2h5( filename, datas, keys ):
    if type(keys) is str:
        sys.exit( 'Write2h5 needs list of datas and keys' )
    with h5.File( filename, 'a' ) as f:
        for data, key in zip( datas, keys ):
            try:
                f.__delitem__( key )
            except:
                pass
            f.create_dataset( key, data=data  )

## read likelihood function from file
def GetLikelihood_IGM( redshift=0., model='primordial', typ='far', nside=64, measure='DM', absolute=True ):
    if redshift < 0.1:
        typ='near'
    if measure == 'DM':
        model='primordial'
    with h5.File( likelihood_file_IGM ) as f:
        P = f[ KeyIGM( redshift=redshift, model=model, typ=typ, nside=nside, measure='|%s|' % measure if absolute else measure, axis='P' ) ].value
        x = f[ KeyIGM( redshift=redshift, model=model, typ=typ, nside=nside, measure='|%s|' % measure if absolute else measure, axis='x' ) ].value
    return P, x



def GetLikelihood_Redshift( population='sfr', telescope='None' ):
    with h5.File( likelihood_file_redshift ) as f:
        P = f[ KeyRedshift( population=population, telescope=telescope, axis='P' ) ].value
        x = f[ KeyRedshift( population=population, telescope=telescope, axis='x' ) ].value
    return P, x

def GetLikelihood_Host_old( redshift=0., model='JF12', weight='uniform', measure='DM' ):
    with h5.File( likelihood_file_galaxy ) as f:
        P = f[ KeyHost( model=model, weight=weight, measure=measure, axis='P' ) ].value * (1+redshift)**( 2 - (measure=='DM') )
        x = f[ KeyHost( model=model, weight=weight, measure=measure, axis='x' ) ].value / (1+redshift)**( 2 - (measure=='DM') )
    return P, x

def GetLikelihood_Host( redshift=0., model='Rodrigues18', weight='smd', measure='DM' ):
    with h5.File( likelihood_file_galaxy ) as f:
        P = f[ KeyHost( model=model, weight=weight, redshift=redshift, measure=measure, axis='P' ) ].value
        x = f[ KeyHost( model=model, weight=weight, redshift=redshift, measure=measure, axis='x' ) ].value
    return P, x


def GetLikelihood_Inter( redshift=0., model='Rodrigues18', measure='DM' ):
    with h5.File( likelihood_file_galaxy ) as f:
        P = f[ KeyInter( redshift=redshift, model=model, measure=measure, axis='P' ) ].value
        x = f[ KeyInter( redshift=redshift, model=model, measure=measure, axis='x' ) ].value
    return P, x

def GetLikelihood_Progenitor( redshift=0., model='Piro18/uniform', measure='DM' ):
    with h5.File( likelihood_file_progenitor ) as f:
        P = f[ KeyProgenitor( model=model, measure=measure, axis='P' ) ].value * (1+redshift)**( 2 - (measure=='DM') )
        x = f[ KeyProgenitor( model=model, measure=measure, axis='x' ) ].value / (1+redshift)**( 2 - (measure=='DM') )
    return P, x

def GetLikelihood_MilkyWay( model='JF12', measure='DM' ):
    with h5.File( likelihood_file_galaxy ) as f:
        P = f[ KeyMilkyWay( model=model, measure=measure, axis='P' ) ].value
        x = f[ KeyMilkyWay( model=model, measure=measure, axis='x' ) ].value
    return P, x


get_likelihood = {
    'IGM'  :       GetLikelihood_IGM,
    'Inter' :      GetLikelihood_Inter,
    'Host' :       GetLikelihood_Host,
    'Progenitor' : GetLikelihood_Progenitor,
    'MilkyWay'   : GetLikelihood_MilkyWay  
}

def GetLikelihood( region, model, density=True, **kwargs ):
    ## wrapper to read any likelihood function written to file
    P, x = get_likelihood[region]( model=model, **kwargs )
    if not density:
        P *= np.diff(x)
    return P, x

def GetLikelihood_Full( redshift=0.1, measure='DM', **scenario ):
    with h5.File( likelihood_file_Full ) as f:
        P = f[ KeyFull( measure=measure, axis='P', redshift=redshift, **scenario ) ].value
        x = f[ KeyFull( measure=measure, axis='x', redshift=redshift, **scenario ) ].value
    return P, x


## Convenient Plot functions
def PlotLikelihood( x, P, density=True, cumulative=False, log=True, ax=None, measure=None, **kwargs ):
    if cumulative:
        density = False
    if ax is None:
        fig, ax = plt.subplots( )
    xx = x[:-1] + np.diff(x)/2
    PP = P * np.diff(x)**(not density) * xx**density
    if cumulative:
        PP = np.cumsum( PP )
    if log:
        ax.loglog()
    ax.plot( xx, PP, **kwargs)

    if measure is not None:
        measure_ = measure if measure=='DM' else '|%s|' % measure
        ax.set_xlabel( 'observed %s / %s' % ( measure_, units[measure] ), fontdict={'size':16 } )
        ax.set_ylabel( ( r"P(%s)" % measure_ ) + ( ( r"$\cdot$%s" % measure_ ) if density else ( r"$\Delta$%s" % measure_ ) ), fontdict={'size':18 } )
#        ax.set_xlabel( measure + ' [%s]' % units[measure], fontdict={'size':20, 'weight':'bold' } )
#        ax.set_ylabel(  'Likelihood', fontdict={'size':24, 'weight':'bold' } )

def Colorbar( x, label=None, labelsize=16, cmap=rainbow ):
    ### plot colorbar at side of plot
    ###  x: 1D array of data to be represented by rainbow colors
    sm = plt.cm.ScalarMappable( cmap=cmap, norm=plt.Normalize(vmin=x.min(), vmax=x.max() ) )
    sm._A = []
    cb = plt.colorbar(sm )
    cb.ax.tick_params( labelsize=labelsize )
    if label is not None:
        cb.set_label(label=label, size=labelsize)


import itertools
def get_steps( N, x, log=False):
    ''' calculate N equal (logarithmic) steps from x[0] to x[1] '''
    if log:
        xx = np.log10(x)
    else:
        xx = x
    x_step = np.linspace( xx[0], xx[1], N)
    if log:
        x_step = 10.**x_step
    return x_step

def mean( x, log=False, **kwargs ):
    ### wrapper to calulate the mean of log-scaled values
    if log:
        return 10.**np.mean( np.log10( x ), **kwargs )
    else:
        return np.mean( x, **kwargs )

def coord2normal(x, lim, log=False):
    ''' transforms coordinate x in (logarithmic) plot to normal coordinates (0,1) '''
    if log:
        return (np.log(x) - np.log(lim[0]))/(np.log(lim[1]) - np.log(lim[0]))
    else:
        return ( x - lim[0] )/( lim[1] - lim[0] )


def plot_limit( ax, x, y, label='', lower_limit=True, arrow_number=2, arrow_length=0.1, arrow_width=0.005, linewidth=4, shift_text_vertical=0, shift_text_horizontal=0 ):
    ### plot upper/lower limit 
    ###  ax: graph to plot limit on
    ###  x,y: one is list of two separate coordinates: define range of limit in one dimension, ons is list of identical coordinates: define limit and limited axis 
    ###  lower_limit=True: plot lower limit, else: plot upper limit
    ###  shift_text_vertical/horizontal: adjust position of text label
    xlog = ax.get_xscale() == 'log'
    ylog = ax.get_yscale() == 'log'
    limit_x, limit_y = int( x[1] == x[0] ), int( y[1] == y[0] )
    upper = -1 if lower_limit else 1
    kwargs = { 'alpha' : 0.7, 'color' : 'gray'}
    plot, = ax.plot( x, y, linestyle='-.', linewidth=linewidth, **kwargs)
    plot.set_dashes([15,5,3,5])
    ax.text(
        mean(x, log=xlog) + shift_text_horizontal, mean(y, log=ylog) + shift_text_vertical,
#        np.mean(y) + upper * limit_x * shift_text_vertical + limit_y * shift_text_horizontal,
        label, fontsize=14, rotation= -90 * limit_x * upper,
        verticalalignment='center', horizontalalignment='center', color=kwargs['color'])
    x_ar, y_ar = get_steps( arrow_number + 2, x, log=xlog)[1:-1], get_steps( arrow_number + 2, y, log=ylog)[1:-1]
    x_length, y_length = - upper * arrow_length * limit_x, - upper * arrow_length * limit_y
#    for xa, ya in itertools.izip( x_ar, y_ar ):
#        ax.arrow( xa, ya, x_length, y_length, width=arrow_width, **kwargs )
    for xa, ya in itertools.izip( coord2normal( x_ar, ax.get_xlim(), log=xlog ), coord2normal( y_ar, ax.get_ylim(), log=ylog ) ):
        plt.arrow( xa, ya, x_length, y_length, transform=ax.transAxes, width=arrow_width, head_width=3*arrow_width, length_includes_head=True, **kwargs )
    return; 


from labels import labels
def LabelAddModel( label, model ):
    ## adds model to label of scenario, i. e. set of combined models
    multi = len(model) > 1
    no = len(model) == 0
    
    label += r"(" * multi
    
    for m in model:
        label += labels[m]
        label += r"+" * multi
    if multi:
        label = label[:-1]
        label += r")"
    label += r"$\ast$" * ( not no )    
    return label


#def LabelScenario( model_Host=[], model_IGM=[], model_Progenitor=[], model_MW=[], weight_Host='' ):
def LabelScenario( **scenario ):
    ## returns plotting label of scenario, i. e. set of combined models
    label = ''
    label = LabelAddModel( label, scenario['model_IGM'] )
    label = LabelAddModel( label, [ m for m in scenario['model_Host'] ] )
    label = LabelAddModel( label, scenario['model_Progenitor'] )
    label = LabelAddModel( label, scenario['model_MW'] )
    return label[:-6]




## mathematical likelihood operations

def histogram( data, bins=10, range=None, density=None, log=False ):
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



def Likelihoods( measurements, P, x, minimial_likelihood=1e-9 ):
    ## returns likelihoods for given measurements according to likelihood function given by P and x
    ## minimal_likelihood is returned for values outside the range of x

    Ps = np.zeros( len( measurements ) ) ## collector for probabilities of measurements
    dx = np.diff(x)
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
                Ps[i_s] = P[i-1]  if i > 0 else minimal_likelihood  ## if that was the lowest bound, probability is ->zero if measurement is outside the range of P, i. e. P~0
                break
        else:
            ## if measure is bigger than the last bin
            Ps[i_s] = minimal_likelihood  ## probability is zero if measurement is outside the range of P, i. e. P~0
    
    return np.array( Ps )


def LikelihoodsAdd( fs, xs, log=True, shrink=False, weights=None, renormalize=1 ):
    ### add together several likelihoos functions
    ###  fs: list of likelihood functions
    ###  xs: list of bin ranges of likelihood functions
    ###  log: set to False if xs are not log-scaled
    ###  shrink=bins: force number of bins in result, otherwise use size of first likelihood function
    ###  weights: provide weights for the likelihood functions
    ### renormalize: total likelihood of the final result

    if len(fs) == 1:
        ## if only one function is given, return the original
        return fs[0], xs[0] 

    ## new function support
    l = len(fs[0])
    if shrink:
        l = shrink
    if log:
        x = 10.**np.linspace( np.log10(np.min(xs)), np.log10(np.max(xs)), l+1 )
    else:
        x = np.linspace( np.min(xs), np.max(xs), l+1 )
    if weights is None:
        weights = np.ones( len(fs) )
        
    P = np.zeros( l )

    ## for each function
    for f, x_f, w in zip(fs, xs, weights):
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
    ##     add average to target likelihood
                P[ib] += w * np.sum( f[ix]*np.diff(x_) ) / (b1-b0)
    if renormalize:
        P *= renormalize/np.sum( P*np.diff(x) )
    return P, x


def LikelihoodShrink( P, x, bins=100, log=True ):
    ### reduce number of bins in likelihood function, contains normalization
    return LikelihoodsAdd( [P], [x], shrink=bins, log=log, renormalize=np.sum( P*np.diff(x) ) )


def LikelihoodConvolve( f, x_f, g, x_g, shrink=True, log=True, absolute=False ):
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
        P = np.sum( [ P[:len(P)/2][::-1], P[len(P)/2:] ], axis=0 )
    ## renormalize full integral to 1
    P /= np.sum( P*np.diff(x) )
    if shrink:
        return LikelihoodShrink( P, x, bins=len(f), log=log )
    else:
        return P, x



def LikelihoodsConvolve( Ps, xs, **kwargs ):
    ### iteratively convolve likelihood functions 
    ###  kwargs for Convole Probability
    P, x = Ps[0], xs[0]
    i = 0.
    for P_, x_ in zip( Ps[1:], xs[1:] ):
        P, x = ConvolveProbability( P, x, P_, x_, **kwargs )
        i += 1
        P /= np.sum( P*np.diff(x) )
    return P, x


def LikelihoodRegion( region, models, weights=None, **kwargs ):
    ### return likelihood for region, if multiple models are provided, their likelihoods are summed together 
    ###  weights: weights to be applied to the models
    ###  kwargs: for GetLikelihood
    Ps, xs = [], []
    for model in models:
        P, x = GetLikelihood( region, model, **kwargs  )
        
        Ps.append( P )
        xs.append( x )
    return AddProbabilities( Ps, xs, weights=weights )


    
def LikelihoodFull( measure='DM', redshift=0.1, nside_IGM=64, force=False, **scenario ):
    ### return the full likelihood function for measure in the given scenario
    ###  redshift: of the source
    ###  nside_IGM: pixelization of IGM full-sky maps
    ###  force=False: Try to read previous results from file, otherwise compute and write to file


    ## check if key is in probability file and return that
    if not force:
        try:
            P, x = GetLikelihood_Full( measure=measure, redshift=np.round(redshift,4), **scenario )
            return P, x
        except:
            pass
                                  
    ## collect likelihood functions for all regions along the LoS
    Ps, xs = [], []
    if len( scenario['model_MW'] ) > 0:
        P, x = LikelihoodRegion( 'MilkyWay', scenario['model_MW'], measure=measure  )
        Ps.append( P )
        xs.append( x )
    if len( scenario['model_IGM'] ) > 0:
        P, x = LikelihoodRegion( 'IGM', scenario['model_IGM'], measure=measure, redshift=redshift, typ='far' if redshift >= 0.1 else 'near', nside=nside_IGM, absolute= measure == 'RM'  )
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
    if len( scenario['model_Progenitor'] ) > 0:
        P, x = LikelihoodRegion( 'Progenitor', scenario['model_Progenitor'], measure=measure, redshift=redshift  )
        Ps.append( P )
        xs.append( x )
    P, x = ConvolveProbabilities( Ps, xs, absolute= measure == 'RM', shrink=True )
    
    ## write to file
    Write2h5( likelihood_file_Full, [P,x], [ KeyFull( measure=measure, redshift=np.round(redshift,4), axis=axis, **scenario ) for axis in ['P','x']] )
    
    return P,x



def LikelihoodMeasureable( min=1., **kwargs ):
    ### returns the part of full likelihood function above the accuracy of telescopes, renormalized to 1
    ###  min: minimal value considered to be measurable
    ###  kwargs: for the full likelihood
    P, x = LikelihoodFull( **kwargs )
    ix, = np.where( x >= min )
    x = x[ix]
    P = P[ix[:-1]] ## remember, x is range of P
    P /= np.sum( P*np.diff(x) )
    return P, x

redshift_bins = np.arange( 0.1,6.1,0.1)
redshift_range = np.arange( 0.0,6.1,0.1)

def LikelihoodRedshift( DMs, scenario, SMs=None, population='flat', telescope='None' ):
    ### returns likelihood functions of redshift for observed DMs (and SMs)
    ### can be used to obtain estimate and deviation
    ###  DMs, SMs: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario: dictionary of models combined to one scenario
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and SMs

    P_redshift = np.zeros( [len(DMs),len(redshift_bins)] )
    ## for each redshift
    for iredshift, redshift in enumerate( redshift_bins ):
        ## calculate the likelihood of observed DM 
        P_redshift[:,iredshift] = Likelihoods( DMs, *LikelihoodFull( typ='DM', redshift=redshift, density=True, **scenario) ) 
    
    ## improve redshift estimate with additional information from SM, which is more sensitive to high overdensities in the LoS
    ## procedure is identical, the likelihood functions are multiplied
    if SMs is not None:
        P_redshift_ = np.zeros( [len(DMs),len(redshift_bins)] )
        for iredshift, redshift in enumerate(redshift_bins):
            P_redshift_[:,iredshift] = Likelihoods( SMs, *LikelihoodFull( typ='SM', redshift=redshift, density=True, **scenario) )  ### not all SM are measureable. However, here we compare different redshifts in the same scenario, so the amount of SM above SM_min is indeed important and does not affect the likelihood of scenarios. Instead, using LikelihoodObservable here would result in wrong estimates.
        P_redshift *= P_redshift_
        P_redshift_= 0
    
    ## consider prior likelihood on redshift according to FRB population and telescope selection effects 
    if population == 'flat':
        pi_redshift = np.array([1.])
    else:
        pi_redshift = GetLikelihood_Redshift( population=population, telescope=telescope )
    P_redshift *= np.resize( pi_redshift, [1,len(redshift_bins)] )
                    
    ## renormalize to 1 for every DM
    P_redshift /= np.resize( np.sum( P_redshift * np.resize( np.diff( redshift_range ), [1,len(redshift_range)-1] ), axis=1 ), [len(DMs),1] )

    return P_redshift, redshift_range

def LikelihoodCombined( DMs, RMs, SMs=None, scenario={}, prior_BO=1., population='flat', telescope='None' ):
    ### compute the likelihood of tuples of DM, RM (and SM) in a LoS scenario
    ###  DMs, RMs, SMs: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario: dictionary of models combined to one scenario
    ###  prior_B0: prior attributed to IGMF model, scalar or 1D array with size identical to DMs
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and SMs


    result = np.zeros( len(DMs) )
    
    ## estimate likelihood of source redshift based on DM and SM
    P_redshift_DMs, redshift_range = LikelihoodRedshift( DMs, scenario, SMs=SMs, population=population, telescope=telescope )
    
    ## for each possible source redshift
    for redshift, p_redshift in zip( redshift_bins, P_redshift_DMs.transpose() ):
        ## estimate likelihood of scenario based on RM, using the redshift likelihood as a prior
        ##  sum results of all possible redshifts
        result += prior_BO * p_redshift * Likelihoods( RMs, *LikelihoodMeasureable( min=RM_min, typ='RM', redshift=redshift, density=False, **scenario) )
    return result



def BayesFactorCombined( DMs, RMs, scenario1, scenario2, SMs=None, population='flat', telescope='None' ):
    ### for set of observed tuples of DM, RM (and SM), compute total Bayes factor that quantifies corroboration towards scenario1 above scenario2 
    ### first computes the Bayes factor for each tuple, then computes the product of all bayes factors
    ###  DMs, RMs, SMs: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario1/2: dictionary of models combined to one scenario
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and SMs
    return np.prod( LikelihoodCombined( DMs, RMs, scenario1, SMs=SMs ) / LikelihoodCombined( DMs, RMs, scenario2, SMs=SMs ) )


def Likelihood2Expectation( P, x, log=True,  density=True ):      ## mean works, std is slightly too high???
    ### computes the estimate value and deviation from likelihood function (must be normalized to 1)
    ###  log: indicates, whether x is log-scaled
    ###  density: indicates whether P is probability density, should always be true
    if log:
        x_log = np.log10(x)
        x_ = x_log[:-1] + np.diff(x_log)/2
    else:
        x_ = x[:-1] + np.diff(x)/2
    if density:
        P_ = P*np.diff(x)
    else:
        P_ = P
    if np.round( np.sum( P_ ), 2) != 1:
        sys.exit( 'P is not normalized' )
    
    x_mean = np.sum( x_*P_ )
    x_std = np.sqrt( np.sum( P_ * ( x_ - x_mean)**2 ) )
    if log:
        x_mean = 10.**x_mean
    return x_mean, x_std


## sample distributions
def uniform_log( lo, hi, N ):
    ## returns N samples of a log-flat distribution from lo to hi
    lo = np.log10(lo)
    hi = np.log10(hi)
    return 10.**np.random.uniform( lo, hi, N )

