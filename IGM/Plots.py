import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Skymaps import GetSkymap
from ProbabilityFunctions import GetProbability
from Parameters import *
from matplotlib.colors import LogNorm
from pylab import cm


def PlotProbability( z, value='DM', typ='near', model=model, nside=nside, density=True, absolute=False, color=None ):
    ## plot probability function at redshift z
    P, x = GetProbability( z, model=model, typ=typ, nside=nside, value=value, absolute=absolute )
    ## if probability density function is provided, plot the probability of each bin, i. e. P * dx
    if density:
        P *= np.diff(x)

    if typ=='near':
        label = 'd = %.0f Mpc' % comoving_radial_distance(0,z).in_units('Mpc')
    else:
        label='z=%.4f' % z
    plt.plot( x[:-1]+np.diff(x)/2, P, label=label, color=color )
    print sum(P*(x[:-1]+np.diff(x)/2))


def PlotProbabilities( value='DM', model=model, typ='near', nside=nside, plot_every=1, absolute=False ):
    ## plots evolution of probability function

    ## determine redshift to be plotted
    if typ == 'near':
        redshifts = redshift_skymaps_near[::plot_every]
    elif typ == 'far':
        redshifts = redshift_skymaps[::plot_every]

    ## plot probability function for each redshift
    color = cm.rainbow( np.linspace( 0, 1, len( redshifts[1:] ) ) )
    for z, c in zip( redshifts[1:], color ):
        PlotProbability( z, value=value, typ=typ, model=model, nside=nside, absolute=absolute, color=c )

    ## care for labels
    plt.ylabel( 'P(%s|%s)' % ( value, 'd' if typ=='near' else 'z' ) )
    plt.xlabel( r"%s (%s)" % ( value, units['RM'] ) )
    if value == 'DM' or absolute:
        plt.xscale( 'log' )
    plt.yscale( 'log' )
    
    plt.legend()

    ## save to file
    plt.savefig( root_probabilities+'%s_%s_%s_probability.png' % ( model, typ, '|%s|' % value if absolute else value ) )
    plt.close()


def PlotSkymap( z, value='DM', typ='near', model=model, nside=nside, min=None, max=None ):
    ## Plot skymap of value at z
    if value == 'DM':
        norm = LogNorm()
        cmap = cm.BuGn_r
    elif value == 'RM':
        norm = LogNorm()
#        norm = None
        cmap = cm.magma

    sky = GetSkymap( z, model=model, typ=typ, value=value, nside=nside )
#    sky_key = '/'.join( [ '/', model, typ, str(nside), value, '%.4f' % z ] )
#    sky = h5.File( skymap_file )[sky_key].value
    if value == 'RM':
        sky = np.abs(sky)
    fig, ax = plt.subplots( subplot_kw={'projection':"mollweide"}, figsize=(8,5) )
    hp.mollview( sky, unit='%s (%s)' % (value,units[value]), title='%s, d = %.0f Mpc' % ( model, comoving_radial_distance(0,z).in_units('Mpc') ), min=min, max=max, norm=norm, cmap=cmap, hold=True )
    

    skyfile = root_skymaps+model+'_%s_z%1.4f.png'
    if not max is None:
        skyfile.replace( '.png', '_m.png' )
    plt.savefig( skyfile % ( value, z ) )
    plt.close()


    
    
def PlotSkymaps( value='DM', typ='near', model=model, nside=nside, min=None, max=None ):
    ## plots skymaps of value for all z, uses same max and min for all

    ## obtain min and max of all skymaps (from first andxo last skymap)
    min_, max_ = np.inf, 0
    for z in [ redshift_skymaps_near[1], redshift_skymaps_near[-1] ]:
        sky_key = '/'.join( [ '/', model, typ, str(nside), value, '%.4f' % z ] )
        sky = h5.File( skymap_file )[sky_key].value
        if value == 'RM':
            sky = np.abs(sky)
        min_ = np.min( [ np.min( sky ), min ] ) if min is None else min
        max_ = np.max( [ np.max( sky ), max ] ) if max is None else max
                      
    for z in redshift_skymaps_near[1:]:
        PlotSkymap( z, value=value, typ=typ, model=model, nside=nside, min=min_, max=max_ )
    return;