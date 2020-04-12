import numpy as np, matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from matplotlib.cm import  YlGn as cmap_gradient
from matplotlib import colors, cm
#from PreFRBLE.convenience import *
from PreFRBLE.label import *
from PreFRBLE.likelihood import *
#from PreFRBLE.physics import *
#from PreFRBLE.parameter import *



############################################################################
############################ PLOT FIGURES ##################################
############################################################################

def PlotBayes( x=np.ones(1), bayes=np.ones(1), title=None, label=None, width=1.0, color='blue', show_values=False, ax=None, posterior=False ):
    """ Barplot of bayes factor or posterior likelihood for values of parameter x """

    if ax is None:
        fig, ax = plt.subplots( )
    ax.bar(x, bayes/bayes.max(), width, color=color )
    ax.set_title( title )
    ax.set_yscale('log')
    ax.set_xlabel( label )
    if posterior:
        ax.set_ylabel(r"$L/L_{\rm max}$")
    else:
        ax.set_ylabel(r"$\mathcal{B}/\mathcal{B}_{\rm max}$")
#        ax.set_ylabel(r"$\mathcal{B} = \prod P / P_0$")
    if show_values: ## print value on top of each bar, .... doesnt work ...
        shift = bayes.max()/bayes.min()/10
        for xx, b in zip( x, bayes ):
            ax.text( xx, b*shift, str(b), color=color, fontweight='bold' )

    ### assure that there are ticks at y axis
    lim = ax.get_ylim()
    ax.set_ylim(lim[0]*0.5, lim[1]*2)

def PlotBayes2D( bayes=[], dev=[], N_bayes=1, x=[], y=[], xlabel='', ylabel='', P_min=1e-5, graphs=False, plane=False, ax=None, posterior=False ):
    """
    Plot 2D distribution of Bayes factors for joint analysis of two parameters x and y
    
    Parameters
    ----------
    x : 1D array-like
        values of first parameter
    y : 1D array-like
        values of second parameter
    bayes : 2D array-like, shape( N_y, N_x )
        bayes factors for tuples of (x,y)
    dev : 2D array-like, shape( N_y, N_x ), optional
        deviation of bayes factors, only plotted for graphs=True
    N_bayes : integer ## depreciated
    graphs : boolean
        indicate whether results should be drawn as graphs 
    plane : boolean
        indicate whether results should be drawn as plane. Do no use together with graphs
    """
    if ax is None:
        fig, ax = plt.subplots()

        
    if posterior:
        P_label = r"L/L_{\rm max}"
    else:
        P_label = r"\mathcal{B}/\mathcal{B}_{\rm max}"
#        P_label = r"\mathcal{B} = \prod P / P_0"
    if graphs:
        for ib, (b, Y) in enumerate( zip( bayes/bayes.max(), y ) ):
            if len(dev) > 0:
                ax.errorbar( x, b, yerr=b*dev[ib], label=Y )
            else:
                ax.plot( x, b, label=Y )
        ax.set_ylabel( "$%s$" % P_label, fontdict={'size':18 }  )
        ax.set_xlabel( xlabel, fontdict={'size':18 }  )
        ax.set_yscale('log')
        ax.legend()
        
    if plane:
        levels = np.linspace( np.log10(P_min), 0, 200 )
#        levels = np.linspace( np.log10(P_min), 0, -np.log10(P_min) )
        colors = Gradient( levels )
        Colorbar( levels, label=r"log$_{10}\left( %s \right)$" % P_label, ax=ax, cmap=cmap_gradient )
        levels = 10.**levels
        
        xx = np.arange( len(x) ) if type(x[0]) is str else x
        yy = np.arange( len(y) ) if type(y[0]) is str else y
        xy_x, xy_y = np.meshgrid( xx, yy )
        
        ax.contourf( xy_x, xy_y, bayes/bayes.max(), levels, colors=colors )
        ax.set_ylabel( ylabel, fontdict={'size':18 }  )
        ax.set_xlabel( xlabel, fontdict={'size':18 }  )
        if type(x[0]) is str:
            ax.set_yticks( xx )
            ax.set_yticklabels( x )
        if type(y[0]) is str:
            ax.set_yticks( yy )
            ax.set_yticklabels( y )



def PlotLikelihood( P=np.ones(1), x=np.arange(2), dev=None, density=True, cumulative=False, log=True, ax=None, measure=None, **kwargs ):
    """
    Plot likelihood function P(x) of measure

    Parameters
    ----------
    dev : array-like, len(P), optional
        give the relative deviation of the plotted likelihood
    cumulative : boolean, 1, -1
        if 1: plot cumulative likelihood starting from lowest x
        if -1: plot cumulative likelihood starting from highest x
        else: plot differential likelihood
    log : boolean
        indicates whether x is log-scaled
    density : boolean
        indicates whether P is probability density, i. e. sum(P*diff(x)) = 1, otherwise assume P is probability, i. e. sum(P) = 1 (Note that renormalization to 1 is not required)
        ### chane to: indicates whether to plot density or probability (P always given as density)
    measure : string
        name of measure x
    **kwargs :  for plt.plot ( or plt.errorbar, if dev is not None )

    """
    if cumulative:
        density = False
    if ax is None:
        fig, ax = plt.subplots( )
    xx = x[:-1] + np.diff(x)/2
    PP = P.copy()
    if log:  ## the probability for a measure to fall in bin depends on size of bin, for logarithmically scaled function, this changes, hence multiply by size of bin
        PP *= np.diff(x)**(not density) * xx**density
    if cumulative:
        PP = np.cumsum( PP )
        if cumulative == -1:
            PP = 1 - PP
    if log:
        ax.loglog()
    if dev is not None:
        ax.errorbar( xx, PP, yerr=dev*PP, **kwargs  )
    else:
        ax.plot( xx, PP, **kwargs)

    if measure is not None:
        ax.set_xlabel( UnitLabel( measure ) , fontdict={'size':16 } )
        ylabel = ( r"P(%s)" % label_measure[measure] ) 
        if cumulative:
            ylabel = r"$\int$"+ylabel+r"${\rm d}$"+label_measure[measure]
        elif log:
            ylabel += ( r"$\times$%s" % label_measure[measure] ) if density else ( r"$\Delta$%s" % label_measure[measure] )
        ax.set_ylabel( ylabel, fontdict={'size':18 } )
#        ax.set_ylabel( ( r"P(%s)" % label_measure[measure] ) + ( ( r"$\times$%s" % label_measure[measure] ) if density else ( r"$\Delta$%s" % label_measure[measure] ) ), fontdict={'size':18 } )
#        ax.set_xlabel( measure + ' [%s]' % units[measure], fontdict={'size':20, 'weight':'bold' } )
#        ax.set_ylabel(  'Likelihood', fontdict={'size':24, 'weight':'bold' } )

def PlotLikelihoodEvolution( measure='DM', dev=False, scenario={}, ax=None, measureable=False, redshift_bins=redshift_bins, colorbar=True, force=False, alpha=0.5, **kwargs ):
    """ 
    Plot likelihood function of measure in different redshift_bins, expected for LoS scenario

    Parameters
    ----------
    dev : boolean
        if True, plot likelihood with deviation 
    measurable : boolean
        if True, plot likelihood only for values accesible to telescope, renormalized to 1
    colorbar : boolean
        if True, plot a colorbar indicating colors for different values of redshift
    force : boolean
        if True, force computation of the full likelihood functions for measure in scenario
    **kwargs : for PlotLikelihood

    """
    if ax is None:
        fig, ax = plt.subplots()
    for z, color in zip( redshift_bins, Rainbow(redshift_bins) ):
        P = GetLikelihood_Full( dev=dev, redshift=z, measure=measure, force=force, **scenario )
        if measureable:
            P = LikelihoodMeasureable( *P, min=measure_range[measure][0], max=measure_range[measure][1] )
        PlotLikelihood( *P, ax=ax, measure=measure, color=color, alpha=alpha, **kwargs )
    if colorbar:
        Colorbar( redshift_bins, label='redshift', ax=ax)


def PlotAverageEstimate( measure='DM', ax=None, scenario={}, errorstart=0, **kwargs ):
    """
    Plot average value of measure as function of redshift. 
    Estimate and deviation are obtained from likelihood function expected in LoS scenario

    Parameter
    ---------
    errorstart : int
        indicate first value for which to plot errorbar
    **kwargs : for ax.errorbar


    """

    if ax is None:
        fig, ax = plt.subplots()

    avg, dev = [], []
    for iz, (redshift, color) in enumerate( zip(redshift_bins, Rainbow(redshift_bins)) ):
        P, x = GetLikelihood_Full( measure=measure, redshift=redshift, **scenario )
        a, s = Likelihood2Expectation( P=P, x=x, density=True, log=True )
        avg.append(a)
        dev.append(s)
    ## plot arrorbars, starting at the indicated position
    erb = ax.errorbar( redshift_bins[errorstart:], avg[errorstart:], np.array(dev).reshape([len(avg),2])[errorstart:].transpose(), **kwargs ) 
    ## draw the full line with the same kwargs
    kwargs_ = kwargs.copy()
    ## however, remove those kwargs that do not work with plt.plot
    for key in ['errorevery', 'label']:
        kwargs_.pop( key, 0 )
    ## if color is not set, ensure that same color is used as for errorbar
    if 'color' not in kwargs:
        lines, collection = erb.get_children()
        color = lines.get_color()
        kwargs_['color'] = color
    ax.plot( redshift_bins, avg, **kwargs_ )
#    ax.errorbar( redshift_bins, avg, avg - 10**(np.log10(avg)-dev), **kwargs ) 
    ax.set_yscale('log')
    ax.set_xlabel('redshift', fontdict={'size':16 })
    ax.set_ylabel('%s / %s' % (label_measure[measure], units[measure]), fontdict={'size':16 } )



def PlotTelescope( measure='DM', measureable=False, telescope='Parkes', population='SMD', ax=None, scenario={}, force=False, dev=True, **kwargs ):
    """ 
    Plot distribution of measure expected to be observed by telescope, assuming a cosmic population and LoS scenario
    
    Parameters
    ----------
    measureable : boolean
       if True, only plot likelihood for values of measure that can actually be observed by telescope
    dev : boolean
       indicate whether to plot deviation of likelihood function
    **kwargs : for PlotLikelihood

    """
    if ax is None:
        fig, ax = plt.subplots()
    P, x, dev_ = GetLikelihood_Telescope(measure=measure, telescope=telescope, population=population, force=force, dev=True, **scenario )
    if measureable:
        P, x, dev_ = LikelihoodMeasureable( P=P, x=x, dev=dev_, min=measure_range[measure][0], max=measure_range[measure][1] )
    PlotLikelihood( x, P, dev=dev_ if dev else [], measure=measure, ax=ax, **kwargs )
    plt.tight_layout()

def PlotContributions( measure='DM', redshift=0.1, **scenario ):
    """ Plot likelihood function for contributions to measure from all regions in scenario for LoS ending at redshift """
    fig, ax = plt.subplots()
    for region in regions:
        models = scenario.get( region )
        if models:
            for model in models:
                P, x = GetLikelihood( region=region, model=model, measure=measure, redshift=redshift )
                PlotLikelihood( x, P, measure=measure, label=region+' '+labels[model] , linestyle=linestyle_region[region], ax=ax )
    plt.legend()
    plt.title( "redshift = %.1f" % redshift )
    plt.tight_layout()


############################################################################
######################## CONVENIENT FUNCTIONS ##############################
############################################################################


def Colorbar( x=np.linspace(0,1,2), label=None, labelsize=16, cmap=rainbow, ax=None ):
    """
    plot colorbar at right side of figure

    Parameter
    ---------
    x : 1D numpy array
        values to be represented by colors
    cmap : cmap object
        determines colorscale used in colorbar
    ax : axis of pyplot.figure, required
        colorbar is added to this axis

    """
    sm = plt.cm.ScalarMappable( cmap=cmap, norm=plt.Normalize(vmin=x.min(), vmax=x.max() ) )
    sm._A = []
    cb = plt.colorbar(sm, ax=ax )
    cb.ax.tick_params( labelsize=labelsize )
    if label is not None:
        cb.set_label(label=label, size=labelsize)

def Rainbow( x=np.linspace(0,1,2), min=None, max=None ):
    """ return rainbow colors for 1D array x """
    if min is None:
        min = x.min()
    if max is None:
        max = x.max()-min
    else:
        max -= min
    x_ = x - min
    x_ /= max
    return rainbow( x_ )

def Gradient( x=np.linspace(0,1,2), min=None, max=None ):
    """ return gradient colors for 1D array x """
    if min is None:
        min = x.min()
    if max is None:
        max = x.max()-min
    else:
        max -= min
    x_ = x - min
    x_ /= max
    return cmap_gradient( x_ )

def AllSidesTicks( ax ):
    """ puts ticks without labels on top and right axis, identical to those on bottom and left axis, respectively """
    axy = ax.twinx()
    axy.set_ylim( ax.get_ylim() )
    axy.set_yscale( ax.get_yscale() )
    axy.set_yticklabels(labels=[])

    axx = ax.twiny()
    axx.set_xlim( ax.get_xlim() )
    axx.set_xscale( ax.get_xscale() )
    axx.set_xticklabels(labels=[])


def get_steps( N=2, x=np.array([1.,10.]), log=False):
    """ calculate N equal (logarithmic) steps from x[0] to x[1] """
    if log:
        xx = np.log10(x)
    else:
        xx = x
    x_step = np.linspace( xx[0], xx[1], N)
    if log:
        x_step = 10.**x_step
    return x_step

def mean( x=10.**np.arange(2), log=False, **kwargs ):
    """ wrapper to calulate the mean of log-scaled values """
    if log:
        return 10.**np.mean( np.log10( x ), **kwargs )
    else:
        return np.mean( x, **kwargs )

def coord2normal(x=10.**np.arange(2), lim=(1,10), log=False):
    """ transforms coordinate x in (logarithmic) plot to normal coordinates (0,1) """
    if log:
        return (np.log10(x) - np.log10(lim[0]))/(np.log10(lim[1]) - np.log10(lim[0]))
    else:
        return ( x - lim[0] )/( lim[1] - lim[0] )


def PlotLimit( ax=None, x=(1,1), y=(1,2), label='', lower_limit=True, arrow_number=2, arrow_length=0.1, arrow_width=0.005, linewidth=4, shift_text_vertical=0, shift_text_horizontal=0 ):
    """
    Plot line with arrows, indicating upper/lower limit 

    Parameter
    ---------
    ax: axis of pyplot.figure (required)
        figure where to plot limit
    x,y: 2-tuple
        indicate range along x and y.
        one has to be a tuple of identical values, indicating the limit on that axis
        the other defines the start and end of the limit line
    lower_limit : boolean
        if True, plot lower limit (arrows pointing to higher values), else: plot upper limit
    shift_text_vertical/horizontal: float
        adjust position of text label along x and y. otherwise placed on center of line. 
        x and y are in unit of the tick values
    """
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
    x_ar = get_steps( arrow_number + 2, coord2normal( x, ax.get_xlim(), log=xlog ))[1:-1]
    y_ar = get_steps( arrow_number + 2, coord2normal( y, ax.get_ylim(), log=ylog ))[1:-1]
    x_length, y_length = - upper * arrow_length * limit_x, - upper * arrow_length * limit_y
#    for xa, ya in itertools.izip( x_ar, y_ar ):
#        ax.arrow( xa, ya, x_length, y_length, width=arrow_width, **kwargs )
#    for xa, ya in itertools.izip( coord2normal( x_ar, ax.get_xlim(), log=xlog ), coord2normal( y_ar, ax.get_ylim(), log=ylog ) ):
    for xa, ya in zip( x_ar, y_ar ):
        ax.arrow( xa, ya, x_length, y_length, transform=ax.transAxes, width=arrow_width, head_width=3*arrow_width, length_includes_head=True, **kwargs )
#        ax.arrow( xa, ya, x_length, y_length, transform=ax.transAxes, width=arrow_width, head_width=3*arrow_width, length_includes_head=True, **kwargs )
    return; 
