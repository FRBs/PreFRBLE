import sys, h5py as h5, numpy as np, yt, csv
from time import time
from PreFRBLE.file_system import *
from PreFRBLE.parameter import *


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



## Read FRBcat

#FRB_dtype = [('ID','S'),('DM','f'),('DM_gal','f'), ('RM','f'),('tau','f'),('host_redshift','S'), ('tele','S')]
FRB_dtype = [('ID','S9'),('DM','f'),('DM_gal','f'), ('RM','S10'),('tau','S10'),('host_redshift','S4'), ('tele','S10')]

def GetFRBcat( telescope=None, RM=None, tau=None, print_number=False ):
    ### read all FRBs from FRBcat
    ###  optional: read only those FRBs observed by telescope with RM and tau
    ###  print_number:True print number of extracted FRBs 
    FRBs = []
    with open( frbcat_file, 'r') as f:
        reader = csv.reader( f )
        header = np.array(next(reader))
        
#        header = np.array(reader.next())
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





