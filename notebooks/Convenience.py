import sys, h5py as h5, numpy as np, matplotlib.pyplot as plt, yt, csv
from time import time
from matplotlib.cm import rainbow
from matplotlib import colors, cm


regions = ['MW', 'IGM', 'Inter', 'Host', 'Local']
linestyle_region = {'MW':'--', 'IGM':'-', 'Inter':":", 'Host':"-.", 'Local':"-."}

models_MW = ['JF12']
models_IGM = ['primordial', 'astrophysical_mean', 'astrophysical_median', 'alpha1-3rd', 'alpha2-3rd', 'alpha3-3rd', 'alpha4-3rd', 'alpha5-3rd', 'alpha6-3rd', 'alpha7-3rd', 'alpha8-3rd', 'alpha9-3rd']
models_Host = ['Rodrigues18/smd', 'Rodrigues18/sfr']
models_Inter = ['Rodrigues18/smd']
models_Local = ['Piro18/uniform/Rodrigues18/smd', 'Piro18/uniform/Rodrigues18/sfr', 'Piro18/wind', 'Piro18/wind+SNR']


telescopes = [ 'ASKAP', 'CHIME', 'Parkes' ]  ## names used in PreFRBLE, identical to telescope names
populations = [ 'SMD', 'SFR', 'coV' ]
telescopes_FRBpoppy = { 'ASKAP':'askap-fly', 'CHIME':'chime', 'Parkes':'parkes' }  ## names used in FRBpoppy
telescopes_FRBcat = { 'ASKAP':'ASKAP', 'CHIME':'CHIME/FRB', 'Parkes':'parkes' }  ## names used in FRBpoppy
populations_FRBpoppy = { 'SFR':'sfr', 'SMD':'smd', 'coV':'vol_co' } ## names used in FRBpoppy


RM_min = 1 # rad m^-2  ## minimal RM measureable by telescopes, is limited by precision of forground removel of Milky Way and Ionosphere
tau_min = 0.01 # ms    ## minimal tau measureable by telescopes, chosen to be smallest value available in FRBcat. However, depends on telescope
tau_max = 50.0 # ms    ## maximal reasonable tau measured by telescopes, chosen to be biggest value observed so far (1906.11305). However, depends on telescope


units = {
    'DM'       :r"pc cm$^{-3}$",
    'RM'       :r"rad m$^{-2}$",
    'SM'       :r"kpc m$^{-20/3}$",
    'tau'      :"ms",
    'z'        :r"z",
    'redshift' :r"1+z",
}

label = {
    'DM'       : 'DM',
    'RM'       : '|RM|',
    'SM'       : 'SM',
    'tau'      : r"$\tau$",
    'z'        :r"z",
    'redshift' :r"1+z",    
}


scale_factor_exponent = { ## used to redshift results of local
    'DM' : 1,
    'RM' : 2,
    'SM' : 2,
    'tau': 3.4
}


## main working folder
root = '/work/stuf315/PreFRBLE/results/'
root = '/hummel/PreFRBLE/'
root = '/data/PreFRBLE/'
root = '/media/hqi/6A57-6B65/PreFRBLE/'

root_likelihood  = root + 'likelihood/'
root_results = root + 'results/'

likelihood_file = root_likelihood+'observables_likelihood.h5'
sky_file = root_results+'observables_maps_galaxy.h5'
frbcat_file = '../frbcat_20191016.csv'

likelihood_file_local = root_likelihood+'observables_likelihood_local.h5'
likelihood_file_galaxy = root_likelihood+'observables_likelihood_galaxy.h5'
likelihood_file_IGM = root_likelihood+'observables_likelihood_IGM.h5'
likelihood_file_redshift = root_likelihood+'redshift_likelihood.h5'

likelihood_file_Full = root_likelihood+'observables_likelihood_Full.h5'
likelihood_file_telescope = root_likelihood+'observables_likelihood_telescope.h5'



## physical constants                                                                                                          
omega_baryon       = 0.048
omega_CDM          = 0.259
omega_matter       = 0.307
omega_lambda       = 0.693
omega_curvature    = 0.0
hubble_constant    = 0.71
 
from yt.units import speed_of_light_cgs as speed_of_light

## cosmic functions
co = yt.utilities.cosmology.Cosmology( hubble_constant=hubble_constant, omega_matter=omega_matter, omega_lambda=omega_lambda, omega_curvature=omega_curvature )
comoving_radial_distance = lambda z0, z1: co.comoving_radial_distance(z0,z1).in_units('Gpc').value

## physics

def AngularDiameterDistance(z_o=0., z_s=1.):
    if type(z_o) is not np.ndarray:
        if type(z_s) is not np.ndarray:
            return ( comoving_radial_distance(0,z_s) - comoving_radial_distance(0,z_o) )/(1+z_s)
        else:
            return np.array([ ( comoving_radial_distance(0,z) - comoving_radial_distance(0,z_o) )/(1+z) for z in z_s.flat])
    else:
        if type(z_s) is not np.ndarray:
            return np.array([ ( comoving_radial_distance(0,z_s) - comoving_radial_distance(0,z) )/(1+z_s) for z in z_o.flat])         
        else:
            return np.array([ ( comoving_radial_distance(0,z2) - comoving_radial_distance(0,z1) )/(1+z2) for z1, z2 in zip( z_o.flat, z_s.flat )])

def Deff( z_s=np.array(1.0), ## source redshift
         z_L=np.array(0.5)   ## redshift of lensing material
        ):
    ### compute ratio of angular diameter distances of lense at redshift z_L for source at redshift z_s
    D_L = AngularDiameterDistance( 0, z_L )
    D_S = AngularDiameterDistance( 0, z_s )
    D_LS = AngularDiameterDistance( z_L, z_s )   
    return D_L * D_LS / D_S


def ScatteringTime( SM=None,  ## kpc m^-20/3, effective SM in the observer frame
                   redshift=0.0, ## of the scattering region, i. e. of effective lense distance
                   D_eff = 1., # Gpc, effective lense distance
                   lambda_0 = 0.23, # m, wavelength
                  ):
    ### computes scattering time in ms of FRB observed at wavelength lambda_0, Marcquart & Koay 2013 Eq.16b 
    return 1.8e5 * lambda_0**4.4 / (1+redshift) * D_eff * SM**1.2
    
def Freq2Lamb( nu=1. ): # Hz 2 meters
    return speed_of_light.in_units('m/s').value / nu

def Lamb2Freq( l=1. ): # meters 2 Hz
    return speed_of_light.in_units('m/s').value / l

HubbleParameter = lambda z: co.hubble_parameter(z).in_cgs()
def HubbleDistance( z ):
    return (speed_of_light / HubbleParameter(z)).in_units('Mpc').value

def PriorInter( z_s=6.0,   ## source redshift
            r=1., ## Mpc, radius of intervening galaxy
            n=1 , ## Mpc^-3 number density of galaxies
            comoving = False ## indicates whether n is comoving
           ):
    ### compute the prior likelihood of galaxies at redshift z to intersect the LoS, integrand of Macquart & Koay 2013 Eq. 33
    z = redshift_bins[redshift_bins<=z_s]
    if (type(n) is not np.ndarray) or comoving:
        ## for comoving number density, consider cosmic expansion
        n = n * (1+z)**3
    return np.pi* r**2 * n * HubbleDistance(z) / (1+z)

def nInter( z_s=6.0,   ## source redshift
            r=1., ## Mpc, radius of intervening galaxy
            n=1 , ## Mpc^-3 number density of galaxies
            comoving = False ## indicates whether n is comoving
           ):
    ### compute the average number of galaxies at redshift z that intersect the LoS, Macquart & Koay 2013 Eq. 33
    dz = np.diff(redshift_range[redshift_range<=z_s*1.000001]) ## small factor required to find correct bin, don't know why it fails without...
    pi_z = PriorInter( z_s, r=r, n=n, comoving=comoving)
    return  pi_z * dz
    

def NInter( z_s=6.,   ## source redshift
            r=1., ## radius of intervening galaxy
            n=1 , ## number density of galaxies
            comoving = False ## indicates whether n is comoving
           ):
    ### returns total intersection likelihood for source at all redshift bins up to z_s
    return np.cumsum( nInter( z_s, r=r, n=n, comoving=comoving) )





## data keys inside likelihood files
def KeyLocal( model='Piro18/wind', measure='DM', axis='P' ):
    return '/'.join( [ model, measure, axis ] )

def KeyMilkyWay( model='JF12', measure='DM', axis='P', redshift=0.0  ):
    return '/'.join( [ 'MilkyWay', model, measure, axis ] )

def KeyHost( redshift=0.0, model='Rodrigues18/smd', measure='DM', axis='P' ):
    return '/'.join( [ 'Host', model, '%.4f' % redshift, measure, axis ] )

def KeyInter( redshift=0.0, model='Rodrigues18', measure='DM', axis='P' ):
    return '/'.join( [ 'Intervening', model, '%.4f' % redshift, measure, axis ] )

def KeyIGM( redshift=0.1, model='primordial', typ='far', nside=2**2, measure='DM', axis='P' ):  ## nside=2**6
    return '/'.join( [ model, typ, str(nside), measure, '%.4f' % redshift, axis ] )

def KeyRedshift( population='flat', telescope='none', axis='P' ):
    return '/'.join( [ population, telescope, axis] )

#def KeyFull( measure='DM', axis='P', redshift=0.1, model_MW=['JF12'], model_IGM=['primordial'], model_Host=['Heesen11/IC10'], weight_Host='StarDensity_MW', model_Local=['Piro18/uniform_JF12'] ):
def KeyFull( measure='DM', axis='P', redshift=0.1, **scenario ):
    models = []
    for region in regions:
        model = scenario.get( region )
        if model:
            models = np.append( models, model )
    models = np.append( models, [ redshift, measure, axis ] )
    return '/'.join( models )

''' old, long and ugly version
    models = np.append( scenario['model_MW'], scenario['model_IGM'] )
    models = np.append( models, scenario['model_Host'] )
    models = np.append( models, scenario['weight_Host'] )
    models = np.append( models, scenario['model_Local'] )
    models = np.append( models, [redshift, measure,axis] )
    return '/'.join( models )
'''

def KeyTelescope( measure='DM', axis='P', telescope='Parkes', population='SMD', **scenario ):
    models = [ telescope, population ]
    for region in regions:
        model = scenario.get( region )
        if model:
            models = np.append( models, model )
    models = np.append( models, [ measure, axis ] )
    return '/'.join( models )


## wrapper to write hdf5 files consistently
def Write2h5( filename='', datas=[], keys=[] ):
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
def GetLikelihood_IGM( redshift=0., model='primordial', typ='far', nside=2**2, measure='DM', absolute=False ):
    if redshift < 0.1:
        typ='near'
    if measure == 'DM':
        model='primordial'
    with h5.File( likelihood_file_IGM ) as f:
        P = f[ KeyIGM( redshift=redshift, model=model, typ=typ, nside=nside, measure='|%s|' % measure if absolute else measure, axis='P' ) ].value
        x = f[ KeyIGM( redshift=redshift, model=model, typ=typ, nside=nside, measure='|%s|' % measure if absolute else measure, axis='x' ) ].value
    return P, x



def GetLikelihood_Redshift( population='SMD', telescope='None' ):
    with h5.File( likelihood_file_redshift ) as f:
        P = f[ KeyRedshift( population=population, telescope=telescope, axis='P' ) ].value
        x = f[ KeyRedshift( population=population, telescope=telescope, axis='x' ) ].value
    return P, x

def GetLikelihood_Host_old( redshift=0., model='JF12', weight='uniform', measure='DM' ):
    with h5.File( likelihood_file_galaxy ) as f:
        P = f[ KeyHost( model=model, weight=weight, measure=measure, axis='P' ) ].value * (1+redshift)**( 2 - (measure=='DM') )
        x = f[ KeyHost( model=model, weight=weight, measure=measure, axis='x' ) ].value / (1+redshift)**( 2 - (measure=='DM') )
    return P, x

def GetLikelihood_Host( redshift=0., model='Rodrigues18/smd', measure='DM' ):
    with h5.File( likelihood_file_galaxy ) as f:
        P = f[ KeyHost( model=model, redshift=redshift, measure=measure, axis='P' ) ].value
        x = f[ KeyHost( model=model, redshift=redshift, measure=measure, axis='x' ) ].value
    return P, x


def GetLikelihood_Inter( redshift=0., model='Rodrigues18', measure='DM' ):
    with h5.File( likelihood_file_galaxy ) as f:
        P = f[ KeyInter( redshift=redshift, model=model, measure=measure, axis='P' ) ].value
        x = f[ KeyInter( redshift=redshift, model=model, measure=measure, axis='x' ) ].value
    return P, x

def GetLikelihood_Local( redshift=0., model='Piro18/uniform', measure='DM' ):
    with h5.File( likelihood_file_local ) as f:
        P = f[ KeyLocal( model=model, measure=measure, axis='P' ) ].value * (1+redshift)**scale_factor_exponent[measure]
        x = f[ KeyLocal( model=model, measure=measure, axis='x' ) ].value / (1+redshift)**scale_factor_exponent[measure]
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
    if not force:
        try:
            with h5.File( likelihood_file_Full ) as f:
                P = f[ KeyFull( measure=measure, axis='P', redshift=redshift, **scenario ) ].value
                x = f[ KeyFull( measure=measure, axis='x', redshift=redshift, **scenario ) ].value
                return P, x
        except:
            pass
    return LikelihoodFull( measure=measure, redshift=redshift, **scenario )

def GetLikelihood_Telescope( telescope='Parkes', population='SMD', measure='DM', force=False, **scenario ):
    if not force:
        try:
            with h5.File( likelihood_file_Full ) as f:
                P = f[ KeyTelescope( telescope=telescope, population=population, measure=measure, axis='P', **scenario ) ].value
                x = f[ KeyTelescope( telescope=telescope, population=population, measure=measure, axis='x', **scenario ) ].value
            return P, x
        except:
            pass
    return LikelihoodTelescope( population=population, telescope=telescope, measure=measure, **scenario )


## Read FRBcat

#FRB_dtype = [('ID','S'),('DM','f'),('DM_gal','f'), ('RM','f'),('tau','f'),('host_redshift','S'), ('tele','S')]
FRB_dtype = [('ID','S9'),('DM','f'),('DM_gal','f'), ('RM','S10'),('tau','S10'),('host_redshift','S4'), ('tele','S10')]

def GetFRBcat( telescope=None, RM=None, tau=None, print_number=False ):
    ### read all FRBs from FRBcat
    ###  optional: read only those FRBs observed by telescope with RM and tau
    ###  print_number:True print number of extracted FRBs 
    FRBs = []
    with open( frbcat_file, 'rb') as f:
        reader = csv.reader( f )
        header = np.array(reader.next())
        i_ID = 0
        i_DM = np.where( header == 'rmp_dm' )[0][0]
        i_DM_gal = np.where( header == 'rop_mw_dm_limit' )[0][0]
        i_RM = np.where( header == 'rmp_rm' )[0][0]
        i_tau = np.where( header == 'rmp_scattering' )[0][0]
        i_zs = np.where( header == 'rmp_redshift_host' )[0][0]
        i_tele = np.where( header == 'telescope' )[0][0]
        i_s = [i_ID, i_DM, i_DM_gal, i_RM, i_tau, i_zs, i_tele]  ## order must fit order of FRB_dtype
        for row in reader:
            if telescope and ( row[i_tele] != telescope_FRBcat[telescope] ) :
                continue
            if tau and ( row[i_tau] == 'null' ) :
                continue
            if RM and ( row[i_RM] == 'null' ) :
                continue
            FRBs.append( tuple( [ row[i].split('&')[0] for i in i_s ] ) )
    if print_number:
        print( len(FRBs) )
    return np.array( FRBs, dtype=FRB_dtype )



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
        ax.set_xlabel( 'observed %s / %s' % ( label[measure], units[measure] ), fontdict={'size':16 } )
        ylabel = ( r"P(%s)" % label[measure] ) 
        ylabel += ( r"$\times$%s" % label[measure] ) if density else ( r"$\Delta$%s" % label[measure] )
        ax.set_ylabel( ylabel, fontdict={'size':18 } )
#        ax.set_ylabel( ( r"P(%s)" % label[measure] ) + ( ( r"$\times$%s" % label[measure] ) if density else ( r"$\Delta$%s" % label[measure] ) ), fontdict={'size':18 } )
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


from labels import labels
def LabelAddModel( label='', model='' ):
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


#def LabelScenario( model_Host=[], model_IGM=[], model_Local=[], model_MW=[], weight_Host='' ):
def LabelScenario( **scenario ):
    ## returns plotting label of scenario, i. e. set of combined models
    label = ''
    for region in regions:
        model = scenario.get( region )
        if model:
            label = LabelAddModel( label, model )
    return label[:-6]

''' old and ugly
    label = LabelAddModel( label, scenario['model_IGM'] )
    label = LabelAddModel( label, [ m for m in scenario['model_Host'] ] )
    label = LabelAddModel( label, scenario['model_Local'] )
    label = LabelAddModel( label, scenario['model_MW'] )
    return label[:-6]
'''



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



def Likelihoods( measurements=[], P=[], x=[], minimal_likelihood=1e-9 ):
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


def LikelihoodsAdd( fs=[], xs=[], log=True, shrink=False, weights=None, renormalize=False ):
    ### add together several likelihoos functions
    ###  fs: list of likelihood functions
    ###  xs: list of bin ranges of likelihood functions
    ###  log: set to False if xs are not log-scaled
    ###  shrink=bins: force number of bins in result, otherwise use size of first likelihood function
    ###  weights: provide weights for the likelihood functions
    ### renormalize: total likelihood of the final result

    if len(fs) == 1:
        ## if only one function is given, return the original
        P, x = fs[0], xs[0] 
        if renormalize:
            P *= renormalize/np.sum( P*np.diff(x) )
        return P, x

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
        P = np.sum( [ P[:len(P)/2][::-1], P[len(P)/2:] ], axis=0 )
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
    Write2h5( likelihood_file_Full, [P,x], [ KeyFull( measure=measure, redshift=np.round(redshift,4), axis=axis, **scenario ) for axis in ['P','x']] )
    
    return P,x

'''   ### old, long and ugly version
    if len( scenario['model_MW'] ) > 0:
        P, x = LikelihoodRegion( region='MW', model=scenario['model_MW'], measure=measure  )
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
    P, x = LikelihoodsAdd( Ps, xs, renormalize=1., weights=Pz ) ### !!! weight Pz, Pz*Dz or Pz*dl(z) ???
    Write2h5( filename=likelihood_file_telescope, datas=[P,x], keys=[ KeyTelescope( measure=measure, telescope=telescope, population=population, axis=axis, **scenario) for axis in ['P','x'] ] )
    return P, x




def LikelihoodMeasureable( min=1., telescope=None, population=None, **kwargs ):
    ### returns the part of full likelihood function above the accuracy of telescopes, renormalized to 1
    ###  min: minimal value considered to be measurable
    ###  kwargs: for the full likelihood
    ###  telescope: indicate survey of telescope to be predicted (requires population. If None, redshift is required)
    if telescope:
        P, x = GetLikelihood_Telescope( telescope=telescope, population=population, **kwargs )
    else:
        P, x = GetLikelihood_Full( **kwargs )

    ix, = np.where( x >= min )
    x = x[ix]
    P = P[ix[:-1]] ## remember, x is range of P
    P /= np.sum( P*np.diff(x) )
    return P, x

redshift_bins = np.arange( 0.1,6.1,0.1)
redshift_range = np.arange( 0.0,6.1,0.1)

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
        Ps[:,iz] = Likelihoods( DMs, *GetLikelihood_Full( typ='DM', redshift=z, density=True, **scenario) ) 
    
    ## improve redshift estimate with additional information from tau, which is more sensitive to high overdensities in the LoS
    ## procedure is identical, the likelihood functions are multiplied
    if taus is not None:
        Ps_ = np.zeros( [len(DMs),len(redshift_bins)] )
        for iz, z in enumerate(redshift_bins):
            Ps_[:,iz] = Likelihoods( taus, *GetLikelihood_Full( typ='tau', redshift=z, density=True, **scenario) )  ### not all tau are measureable. However, here we compare different redshifts in the same scenario, so the amount of tau above tau_min is indeed important and does not affect the likelihood of scenarios. Instead, using LikelihoodObservable here would result in wrong estimates.
        Ps *= Ps_
        Ps_= 0
    
    ## consider prior likelihood on redshift according to FRB population and telescope selection effects 
    if population == 'flat':
        pi = np.array([1.])
    else:
        pi, x = GetLikelihood_Redshift( population=population, telescope=telescope )
    Ps = Ps * np.resize( pi, [1,len(redshift_bins)] )
                    
    ## renormalize to 1 for every DM
    Ps = Ps / np.resize( np.sum( Ps * np.resize( np.diff( redshift_range ), [1,len(redshift_bins)] ), axis=1 ), [len(DMs),1] )

    return Ps, redshift_range

def LikelihoodCombined( DMs=[], RMs=[], taus=None, scenario={}, prior_BO=1., population='flat', telescope='None' ):
    ### compute the likelihood of tuples of DM, RM (and tau) in a LoS scenario
    ###  DMs, RMs, taus: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario: dictionary of models combined to one scenario
    ###  prior_B0: prior attributed to IGMF model, scalar or 1D array with size identical to DMs
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and taus


    result = np.zeros( len(DMs) )
    
    ## estimate likelihood of source redshift based on DM and tau
    P_redshifts_DMs, redshift_range = LikelihoodRedshift( DMs=DMs, scenario=scenario, taus=taus, population=population, telescope=telescope )
    
    ## for each possible source redshift
    for redshift, P_redshift in zip( redshift_bins, P_redshifts_DMs.transpose() ):
        ## estimate likelihood of scenario based on RM, using the redshift likelihood as a prior
        ##  sum results of all possible redshifts
        P, x = LikelihoodMeasureable( min=RM_min, typ='RM', redshift=redshift, density=False, **scenario )
        result += prior_BO * P_redshift * Likelihoods( measurements=RMs, P=P, x=x )
 
    return result



def BayesFactorCombined( DMs=[], RMs=[], scenario1={}, scenario2={}, taus=None, population='flat', telescope='None' ):
    ### for set of observed tuples of DM, RM (and tau), compute total Bayes factor that quantifies corroboration towards scenario1 above scenario2 
    ### first computes the Bayes factor for each tuple, then computes the product of all bayes factors
    ###  DMs, RMs, taus: 1D arrays of identical size, contain extragalactic component of observed values
    ###  scenario1/2: dictionary of models combined to one scenario
    ###  population: assumed cosmic population of FRBs
    ###  telescope: in action to observe DMs, RMs and taus
    L1 = LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario1, taus=taus )
    L2 = LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario2, taus=taus )
    B = np.prod(L1/L2)
#    print( L1 )
#    print( L2 )
#    print( B )
    return np.prod( LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario1, taus=taus ) / LikelihoodCombined( DMs=DMs, RMs=RMs, scenario=scenario2, taus=taus ) )


def Likelihood2Expectation( P=np.array(0), x=np.array(0), log=True,  density=True ):      ## mean works, std is slightly too high???
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

def RandomSample( N=1, P=np.array(0), x=np.array(0), log=True ):
    ### return sample of N according to likelihood function P(x) 
    ###  P is renormalized probability density, i. e. sum(P*dx)=1
    ###  log: indicates whether x is log-scaled
    Pd = P*np.diff(x)
    if np.round( np.sum(Pd), 4) != 1:
        sys.exit( " 1 != %f" % (np.sum(Pd)) )
    f = Pd.max()
    lo, hi = x[0], x[-1]
    if log:
        lo, hi = np.log10( [lo,hi] )
    res = []
    while len(res) < N:
        r = np.random.uniform( high=hi, low=lo, size=N )
        if log:
            r = 10.**r
        z = np.random.uniform( size=N )
        p = Likelihoods( r, P/f, x )
        res.extend( r[ np.where( z < p )[0] ] )
    return res[:N]


def FakeFRBs( measures=['DM','RM'], N=50, telescope='CHIME', population='SMD', **scenario):
    ### returns measures of a fake survey of N FRBs expected to be observed by telescope assuming population & scenario for LoS
    FRBs = {}
    for measure in measures:
        ## load likelihood function 
        if measure == 'RM':
            ## due to the ionosphere foreground, only allow for RM > 1 rad m^-2 to be observed
            P, x = LikelihoodMeasureable( min=RM_min, measure=measure, telescope=telescope, population=population, **scenario )
        else:
            P, x = GetLikelihood_Telescope( measure=measure, telescope=telescope, population=population, **scenario )
    
        ##   sample likelihood function
        FRBs[measure] = RandomSample( N, P, x )
    
    return FRBs

def uniform_log( lo=1., hi=2., N=10 ):
    ## returns N samples of a log-flat distribution from lo to hi
    lo = np.log10(lo)
    hi = np.log10(hi)
    return 10.**np.random.uniform( lo, hi, N )


