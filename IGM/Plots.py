import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Skymaps import GetSkymap
from Physics import comoving_radial_distance
from LikelihoodFunctions import GetLikelihood
from Parameters import *
from matplotlib.colors import LogNorm
from pylab import cm



### !!! not needed, double ??? (notebook)
def PlotLikelihood( z, measure='DM', typ='near', model=model, nside=nside, density=True, absolute=False, color=None ):
    ## plot likelihood function at redshift z
    P, x = GetLikelihood( z, model=model, typ=typ, nside=nside, measure=measure, absolute=absolute )
    print 'likelihood renormalization check 1 =', np.sum( P *  np.diff(x) )
    if density:
        ## plot the probability of each bin, i. e. P * dx
        P *= np.diff(x)

    if typ=='near':
        label = 'd = %.0f Mpc' % comoving_radial_distance(0,z).in_units('Mpc')
    else:
        label='z=%.4f' % z
    plt.plot( x[:-1] + np.diff(x) / 2, P, label=label, color=color )

### !!! not needed, double ??? (notebook)
def PlotLikelihoods( measure='DM', model=model, typ='near', nside=nside, plot_every=1, absolute=False ):
    ## plots evolution of likelihood function with z

    ## determine redshift to be plotted
    if typ == 'near':
        redshifts = redshift_skymaps_near[::plot_every]
    elif typ == 'far':
        redshifts = redshift_skymaps[::plot_every]

    ## plot likelihood function for each redshift
    color = cm.rainbow( np.linspace( 0, 1, len( redshifts[1:] ) ) )
    for z, c in zip( redshifts[1:], color ):
        PlotLikelihood( z, measure=measure, typ=typ, model=model, nside=nside, absolute=absolute, color=c )

    ## care for labels
    plt.ylabel( 'P(%s|%s)' % ( measure, 'd' if typ=='near' else 'z' ) )
    plt.xlabel( r"%s (%s)" % ( measure, units[measure] ) )
    if measure in [ 'DM', 'SM' ] or absolute:
        plt.xscale( 'log' )
    plt.yscale( 'log' )
    
    plt.legend()

    ## save to file
    plt.savefig( root_likelihoods+'%s_%s_%s_likelihood.png' % ( model, typ, '|%s|' % measure if absolute else measure ) )
    plt.close()


def PlotSkymap( z, measure='DM', typ='near', model=model, nside=nside, min=None, max=None ):
    ## Plot skymap of measure at z

    ## choose color scale for observable
    if measure == 'DM':
        cmap = cm.BuGn_r
    elif measure == 'RM':
        cmap = cm.magma
    elif measure == 'SM':
        cmap = cm.YlGnBu_r

    ## read skymap from skymap_file
    sky = GetSkymap( z, model=model, typ=typ, measure=measure, nside=nside )
    if measure == 'RM':
        sky = np.abs(sky)
    fig, ax = plt.subplots( subplot_kw={'projection':"mollweide"}, figsize=(8,5) )
    hp.mollview( sky, unit='%s / (%s)' % (measure,units[measure]), title='%s, d = %.0f Mpc' % ( model, comoving_radial_distance(0,z).in_units('Mpc') ), min=min, max=max, norm = LogNorm(), cmap=cmap, hold=True )
    

    skyfile = root_skymaps+model+'_%s_z%1.4f.png'
    if not max is None:
        skyfile.replace( '.png', '_m.png' )
    plt.savefig( skyfile % ( measure, z ) )
    plt.close()


    
    
def PlotSkymaps( measure='DM', typ='near', model=model, nside=nside, min=None, max=None ):
    ## plots skymaps of measure for all z, uses same max and min for all

    ## obtain min and max of all skymaps (from first andxo last skymap)
    min_, max_ = np.inf, 0
    for z in [ redshift_skymaps_near[1], redshift_skymaps_near[-1] ]:
        sky_key = '/'.join( [ '/', model, typ, str(nside), measure, '%.4f' % z ] )
        sky = h5.File( skymap_file )[sky_key].value
        if measure == 'RM':
            sky = np.abs(sky)
        min_ = np.min( [ np.min( sky ), min ] ) if min is None else min
        max_ = np.max( [ np.max( sky ), max ] ) if max is None else max
                      
    for z in redshift_skymaps_near[1:]:
        PlotSkymap( z, measure=measure, typ=typ, model=model, nside=nside, min=min_, max=max_ )
    return;



def PlotNearRays( measure='DM', nside=nside, model=model ):
    with h5.File( skymap_file ) as f:
        Ms = []
        zs = f['%s/near/%i/%s' % ( model, nside, measure) ].keys()
        for z in zs:
            Ms.append( f['%s/near/%i/%s/%s' % ( model, nside, measure,z)] .value )
    Ms = np.array( Ms ).transpose()
    zs = np.array( zs, dtype='float' )
    for M in Ms:
        plt.plot( zs, M )
    plt.yscale('log')
    plt.ylabel( '%s / (%s) ' % (measure, units[measure] ) )
    plt.xlabel('redshift' )
    plt.savefig( root_rays + "%s_redshift_near_%s.png" % ( measure, model ) )
    plt.close()


import yt

def PlotFarRays( measure='DM', nside=nside, model=model, plot_mean=True, plot_stddev=True, save_mean=True, uniform=False, plot_single_rays=False, overestimate_SM=False ):
    Ms = []
    with h5.File( LoS_observables_file ) as f:
        for i in f['%s/chopped/' % model].keys():
            M = f['%s/chopped/%s/%s' % ( model, i, measure + 'overestimate'*overestimate_SM ) ].value
            if measure == 'SM':
                M *= kpc2cm/100   * 1e-12  ## to match units in Zhu et al. !!! remove
            Ms.append( M )
            if plot_single_rays:
                plt.plot( np.arange(0.1,6.1,0.1) , M )
    if plot_single_rays:
        plt.yscale('log')
        if measure == 'SM':
            plt.ylim(2e0,2e5)
        plt.xlim(0,2)
        plt.ylabel( '%s / (%s) ' % ( measure, units[measure] ) )
        plt.xlabel( 'redshift' )
        plt.savefig( root_rays + "%s_redshift_%s.png" % ( measure, model ) )
        plt.close()

    if plot_mean: 
        zs = np.arange(0.0,6.1,0.1)
        if measure == 'RM':
            Ms = np.abs(Ms)
##        Ms = np.cumsum( Ms, axis=1 )  #### !!!! double cumsum?? results are written as cumsum
#        plt.errorbar( np.arange(0.1,6.1,0.1) , np.mean(Ms, axis=0), np.std(Ms, axis=0 ) )
        Ms_mean, Ms_std = np.mean(Ms, axis=0), np.std(Ms, axis=0 )
        plt.plot( zs[1:] , Ms_mean, label='mean ' + 'overestimate '*overestimate_SM +model, linewidth=2 )
        if plot_stddev:
            plt.plot( zs[1:] , Ms_mean + Ms_std, linestyle=':', label='stddev' )
        if uniform:
            ## print prediction for homogeneous ICM
            zs = np.arange(0.0,6.1,0.01)  ## 0.01 leads to 3% accuracy of results
            dl = np.array([ luminosity_distance( z0, z1 ) for z0, z1 in zip( zs, zs[1:] ) ])
            if measure == 'SM':
                '''
                SM_uniform = ScatteringMeasure( density=1, overdensity=True, outer_scale=1., redshift=zs[1:], distance=dl )
                SM_uniform *= kpc2cm/100*1e-12
                plt.plot( zs[1:], np.cumsum(SM_uniform), linestyle='--', label='diffuse dl' )
                '''
                ## uniform prediction for cosmology parameters used in Hackstein et al. 2019
                SM_uniform = ScatteringMeasure_ZHU( density=1, overdensity=True, outer_scale=1., redshift=zs )
                SM_uniform *= kpc2cm/100*1e-12  ## to match units in Zhu et al. !!! remove
                plt.plot( zs[1:], np.cumsum(SM_uniform), linestyle='-.', label='diffuse' )

                ## mimic results of Zhu et al. 
                SM_uniform = ScatteringMeasure_ZHU( density=1, overdensity=True, outer_scale=1., redshift=zs, omega_matter=0.317, omega_lambda=0.683, omega_baryon=0.049, hubble_constant=0.671 )
                SM_uniform *= kpc2cm/100*1e-12  ## to match units in Zhu et al. !!! remove
                plt.plot( zs[1:], np.cumsum(SM_uniform), linestyle='-.', label='diffuse, Zhu+18' )

                ## mimic results of Macquart & Koay 2013 
                SM_uniform = ScatteringMeasure_ZHU( density=1, overdensity=True, outer_scale=1., redshift=zs, omega_matter=0.3, omega_lambda=0.7, omega_baryon=0.04, hubble_constant=0.71 )
                SM_uniform *= kpc2cm/100*1e-12  ## to match units in Zhu et al. !!! remove
                plt.plot( zs[1:], np.cumsum(SM_uniform), linestyle='-.', label='diffuse, MK13' )

            plt.legend()

        plt.yscale('log')
        plt.ylabel( '%s / (%s) ' % ( measure, units[measure] ) )
        plt.xlabel( 'redshift' )
        if measure == 'SM':
            plt.ylim(2e12,2e17)
            plt.ylim(2e0,2e5)
        plt.xlim(0,2)
#            plt.ylim(2e10,2e17)
        if save_mean:
            plt.savefig( root_rays + "%s_mean_redshift_%s.png" % ( measure, model ) )
            plt.close()


    
