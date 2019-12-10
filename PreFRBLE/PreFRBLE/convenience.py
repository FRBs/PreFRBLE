import sys, h5py as h5, numpy as np, yt, csv
from time import time
from PreFRBLE.file_system import *
#from PreFRBLE.physics import *
from PreFRBLE.labels import labels

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
        P = f[ KeyHost( model=model, weight=weight, measure=measure, axis='P' ) ].value * (1+redshift)**scale_factor_exponent[measure]
        x = f[ KeyHost( model=model, weight=weight, measure=measure, axis='x' ) ].value / (1+redshift)**scale_factor_exponent[measure]
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

    if len(scenario) == 1:
        region, model = scenario.copy().popitem()
        return GetLikelihood( region=region, model=model, redshift=redshift, measure=measure )
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



## labelling functions

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


