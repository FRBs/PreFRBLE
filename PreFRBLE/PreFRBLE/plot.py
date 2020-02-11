import numpy as np, matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from matplotlib.cm import  YlGn as cmap_gradient
from matplotlib import colors, cm
#from PreFRBLE.convenience import *
from PreFRBLE.label import *
from PreFRBLE.likelihood import *
#from PreFRBLE.physics import *
#from PreFRBLE.parameter import *



## Convenient Plot functions
def PlotBayes( x=np.ones(1), y=np.ones(1), title=None, label=None, width=1.0, color='blue', show_values=False, ax=None, posterior=False ):
    if ax is None:
        fig, ax = plt.subplots( )
    ax.bar(x, y/y.max(), width, color=color )
    ax.set_title( title )
    ax.set_yscale('log')
    ax.set_xlabel( label )
    if posterior:
        ax.set_ylabel(r"$L/L_{\rm max}$")
    else:
        ax.set_ylabel(r"$\mathcal{B}/\mathcal{B}_{\rm max}$")
#        ax.set_ylabel(r"$\mathcal{B} = \prod P / P_0$")
    if show_values: ## print value on top of each bar, .... doesnt work ...
        shift = y.max()/y.min()/10
        for xx, yy in zip( x, y ):
            ax.text( xx, yy*shift, str(yy), color=color, fontweight='bold' )

    ### assure that there are ticks at y axis
    lim = ax.get_ylim()
    ax.set_ylim(lim[0]*0.5, lim[1]*2)

def PlotBayes2D( bayes=[], N_bayes=1, x=[], y=[], xlabel='', ylabel='', P_min=1e-5, graphs=False, plane=False, ax=None, posterior=False ):
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
    N_bayes : integer
        number of events that enter bayes. Used to plot Poisson noise. ## depreciated, since Poisson noise is misleading error estimate for bayes
    graphs : boolean
        indicate whether results should be drawn as graphs 
    plane : boolean
        indicate whether results should be drawn as plane
    """
    if ax is None:
        fig, ax = plt.subplots()

        
    if posterior:
        P_label = r"L/L_{\rm max}"
    else:
        P_label = r"\mathcal{B}/\mathcal{B}_{\rm max}"
#        P_label = r"\mathcal{B} = \prod P / P_0"
    if graphs:
#        noise =  N_bayes**-0.5 if N_bayes > 1 else 0
        for b, Y in zip( bayes/bayes.max(), y ):
            ax.plot( x, b, label=Y )
#            ax.errorbar( x, b, yerr=b*noise, label=Y )
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



def PlotLikelihood( x=np.arange(2), P=np.ones(1), density=True, cumulative=False, log=True, ax=None, measure=None, **kwargs ):
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

def PlotLikelihoodEvolution( measure='DM', scenario={}, ax=None, measureable=False, redshift_bins=redshift_bins, colorbar=True, force=False, **kwargs ):
    if ax is None:
        fig, ax = plt.subplots()
    for z, color in zip( redshift_bins, Rainbow(redshift_bins) ):
        P, x = GetLikelihood_Full( redshift=z, measure=measure, force=force, **scenario )
        if measureable:
            P, x = LikelihoodMeasureable( P=P, x=x, min=measure_range[measure][0], max=measure_range[measure][1] )
        PlotLikelihood(P=P, x=x, ax=ax, measure=measure, color=color, **kwargs )
    if colorbar:
        Colorbar( redshift_bins, label='redshift', ax=ax)


def PlotAverageEstimate( measure='DM', ax=None, scenario={}, errorstart=0, **kwargs ):
    if ax is None:
        fig, ax = plt.subplots()

    avg, std = [], []
    for iz, (redshift, color) in enumerate( zip(redshift_bins, Rainbow(redshift_bins)) ):
        P, x = GetLikelihood_Full( measure=measure, redshift=redshift, **scenario )
        a, s = Likelihood2Expectation( P=P, x=x, density=True, log=True )
        avg.append(a)
        std.append(s)
    ## plot arrorbars, starting at the indicated position
    erb = ax.errorbar( redshift_bins[errorstart:], avg[errorstart:], np.array(std).reshape([len(avg),2])[errorstart:].transpose(), **kwargs ) 
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
#    ax.errorbar( redshift_bins, avg, avg - 10**(np.log10(avg)-std), **kwargs ) 
    ax.set_yscale('log')
    ax.set_xlabel('redshift', fontdict={'size':16 })
    ax.set_ylabel('%s / %s' % (label_measure[measure], units[measure]), fontdict={'size':16 } )



def PlotTelescope( measure='DM', measureable=False, telescope='Parkes', population='SMD', ax=None, scenario={}, force=False, **kwargs ):
    ### Plot distribution of measure expected to be observed by telescope, assuming a cosmic population and LoS scenario
    if ax is None:
        fig, ax = plt.subplots()
    P, x = GetLikelihood_Telescope(measure=measure, telescope=telescope, population=population, force=force, **scenario )
    if measureable:
        P, x = LikelihoodMeasureable( P=P, x=x, min=measure_range[measure][0], max=measure_range[measure][1] )
    PlotLikelihood( x, P, measure=measure, ax=ax, **kwargs )
    plt.tight_layout()

def PlotContributions( measure='DM', redshift=0.1, **scenario ):
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

def Colorbar( x=np.linspace(0,1,2), label=None, labelsize=16, cmap=rainbow, ax=None ):
    ### plot colorbar at side of plot
    ###  x: 1D array of data to be represented by rainbow colors
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
    """ return rainbow colors for 1D array x """
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
    ''' calculate N equal (logarithmic) steps from x[0] to x[1] '''
    if log:
        xx = np.log10(x)
    else:
        xx = x
    x_step = np.linspace( xx[0], xx[1], N)
    if log:
        x_step = 10.**x_step
    return x_step

def mean( x=10.**np.arange(2), log=False, **kwargs ):
    ### wrapper to calulate the mean of log-scaled values
    if log:
        return 10.**np.mean( np.log10( x ), **kwargs )
    else:
        return np.mean( x, **kwargs )

def coord2normal(x=10.**np.arange(2), lim=(1,10), log=False):
    ''' transforms coordinate x in (logarithmic) plot to normal coordinates (0,1) '''
    if log:
        return (np.log10(x) - np.log10(lim[0]))/(np.log10(lim[1]) - np.log10(lim[0]))
    else:
        return ( x - lim[0] )/( lim[1] - lim[0] )


def PlotLimit( ax=None, x=[1,1], y=[1,2], label='', lower_limit=True, arrow_number=2, arrow_length=0.1, arrow_width=0.005, linewidth=4, shift_text_vertical=0, shift_text_horizontal=0 ):
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
