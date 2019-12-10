import numpy as np, matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from matplotlib import colors, cm

## Convenient Plot functions
def PlotLikelihood( x=np.arange(2), P=np.ones(1), density=True, cumulative=False, log=True, ax=None, measure=None, **kwargs ):
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
        ax.set_xlabel( 'observed %s / %s' % ( label_measure[measure], units[measure] ), fontdict={'size':16 } )
        ylabel = ( r"P(%s)" % label_measure[measure] ) 
        ylabel += ( r"$\times$%s" % label_measure[measure] ) if density else ( r"$\Delta$%s" % label_measure[measure] )
        ax.set_ylabel( ylabel, fontdict={'size':18 } )
#        ax.set_ylabel( ( r"P(%s)" % label_measure[measure] ) + ( ( r"$\times$%s" % label_measure[measure] ) if density else ( r"$\Delta$%s" % label_measure[measure] ) ), fontdict={'size':18 } )
#        ax.set_xlabel( measure + ' [%s]' % units[measure], fontdict={'size':20, 'weight':'bold' } )
#        ax.set_ylabel(  'Likelihood', fontdict={'size':24, 'weight':'bold' } )

def PlotTelescope( measure='DM', telescope='Parkes', population='SMD', ax=None, label=None, scenario={}, **kwargs ):
    ### Plot distribution of measure expected to be observed by telescope, assuming a cosmic population and LoS scenario
    if ax is None:
        fig, ax = plt.subplots()
    P, x = GetLikelihood_Telescope(measure=measure, telescope=telescope, population=population, **scenario )
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

def Rainbow( x=np.linspace(0,1,2) ):
    ### return rainbow colors for 1D array x
    x_ = x - x.min()
    x_ /= x_.max()
    return rainbow( x_ )


import itertools
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
    for xa, ya in itertools.izip( x_ar, y_ar ):
        ax.arrow( xa, ya, x_length, y_length, transform=ax.transAxes, width=arrow_width, head_width=3*arrow_width, length_includes_head=True, **kwargs )
#        ax.arrow( xa, ya, x_length, y_length, transform=ax.transAxes, width=arrow_width, head_width=3*arrow_width, length_includes_head=True, **kwargs )
    return; 
