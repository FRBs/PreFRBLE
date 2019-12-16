from __future__ import print_function
import sys, h5py as h5, numpy as np, yt, csv
from time import time, sleep
from PreFRBLE.file_system import *
from PreFRBLE.parameter import *



def CorrectScenario( measure='DM', **scenario ):
    ## this function is used to correct scenario keys wenn reading data, since some models have output stored under different name
    
    result = scenario.copy()
    
    ## combine primordial and alpha results, former for DM, SM and tau, latter for RM (is identical to primordial, which was also computed)
    if 'RM' in measure:
        pass
    ## rhis is not needed right now
    #    if scenario['IGM'] == ['primordial']:
    #        result['IGM'] = ['alpha1-3rd']
    else:
        if 'alpha' in scenario['IGM'][0]:
            result['IGM'] = ['primordial']
    return result


## data keys inside likelihood files
def KeyLocal( model='Piro18/wind', measure='DM', axis='P' ):
    return '/'.join( [ model, measure, axis ] )

def KeyMilkyWay( model='JF12', measure='DM', axis='P', redshift=0.0  ):
    return '/'.join( [ 'MilkyWay', model, measure, axis ] )

def KeyHost( redshift=0.0, model='Rodrigues18/smd', measure='DM', axis='P' ):
    return '/'.join( [ 'Host', model, '%.4f' % np.round( redshift, redshift_accuracy ), measure, axis ] )

def KeyInter( redshift=0.0, model='Rodrigues18', measure='DM', axis='P' ):
    return '/'.join( [ 'Intervening', model, '%.4f' % np.round( redshift, redshift_accuracy ), measure, axis ] )

def KeyIGM( redshift=0.1, model='primordial', typ='far', nside=2**2, measure='DM', axis='P' ):  ## nside=2**6
#    print( measure, model, redshift )
    model_ = CorrectScenario( measure=measure, IGM=[model] )['IGM'][0]
#    print(model_)
    return '/'.join( [ model_, typ, str(nside), measure, '%.4f' % np.round( redshift, redshift_accuracy ), axis ] )

def KeyRedshift( population='flat', telescope='none', axis='P' ):
    return '/'.join( [ population, telescope, axis] )

#def KeyFull( measure='DM', axis='P', redshift=0.1, model_MW=['JF12'], model_IGM=['primordial'], model_Host=['Heesen11/IC10'], weight_Host='StarDensity_MW', model_Local=['Piro18/uniform_JF12'] ):
def KeyFull( measure='DM', axis='P', redshift=0.1, **scenario ):
    scenario_ = CorrectScenario( measure, **scenario )
    models = []
    for region in regions:
        model = scenario_.get( region )
        if model:
            models = np.append( models, model )
    models = np.append( models, [ np.round( redshift, redshift_accuracy ), measure, axis ] )
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
    scenario_ = CorrectScenario( measure, **scenario )
    models = [ telescope, population ]
    for region in regions:
        model = scenario_.get( region )
        if model:
            models = np.append( models, model )
    models = np.append( models, [ measure, axis ] )
    return '/'.join( models )


## wrapper to write hdf5 files consistently
def Write2h5( filename='', datas=[], keys=[] ):
    if type(keys) is str:
        sys.exit( 'Write2h5 needs list of datas and keys' )
    with SimpleFlock( 'flock_writeh5', 1 ):  ## to prevent multiple processes to write to file simultaneously (which will fail)
#    with SimpleFlock( filename, 1 ):  ## to prevent multiple processes to write to file simultaneously (which will fail)
        with h5.File( filename, 'a' ) as f:
            for data, key in zip( datas, keys ):
                try:
                    f.__delitem__( key )
                except:
                    pass
                f.create_dataset( key, data=data  )



## Read FRBcat

#FRB_dtype = [('ID','S'),('DM','f'),('DM_gal','f'), ('RM','f'),('tau','f'),('host_redshift','S'), ('tele','S')]
#FRB_dtype = [('ID','U9'),('DM','f'),('DM_gal','f'), ('RM','U10'),('tau','U10'),('host_redshift','U4'), ('tele','U10')]
FRB_dtype = [('ID','U9'),('DM','f'),('DM_gal','f'), ('RM','f'),('tau','f'),('host_redshift','f'), ('tele','U10')]

def GetFRBcat( telescopes=None, RM=None, tau=None, print_number=False ):
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
            if telescopes and ( row[i_tele] not in [telescopes_FRBcat[tele] for tele in telescopes] ) :
                continue
            if tau and ( row[i_tau] == 'null' ) :
                continue
            if RM and ( row[i_RM] == 'null' ) :
                continue
            FRBs.append( tuple( [ decode(row[i].split('&')[0], dtype) for i, dtype in zip( i_s, np.array(FRB_dtype)[:,1] ) ] ) )
    return np.array( FRBs, dtype=FRB_dtype )


## short wrapper to decode byte-strings read from FRBcat
def decode( string, dtype='U' ):
    if 'f' in dtype:
        if 'null' in string:
            return float('NaN')
        return float(string)
    return string




## flocker to keep parallel processes from writing to same file simultaneously
## provided by derpston, https://github.com/derpston/python-simpleflock/blob/master/src/simpleflock.py#L14

import os, fcntl, errno

class SimpleFlock:
   """Provides the simplest possible interface to flock-based file locking. Intended for use with the `with` syntax. It will create/truncate/delete the lock file as necessary."""

   def __init__(self, path, timeout = None):
      self._path = path
      self._timeout = timeout
      self._fd = None

   def __enter__(self):
      self._fd = os.open(self._path, os.O_CREAT)
      start_lock_search = time()
      while True:
         try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Lock acquired!
            return
         except (OSError, IOError) as ex:
            if ex.errno != errno.EAGAIN: # Resource temporarily unavailable
               raise
            elif self._timeout is not None and time() > (start_lock_search + self._timeout):
               # Exceeded the user-specified timeout.
               print( "timeout exceeded" ) 
               raise
         
         # TODO It would be nice to avoid an arbitrary sleep here, but spinning
         # without a delay is also undesirable.
         sleep(0.1)

   def __exit__(self, *args):
      fcntl.flock(self._fd, fcntl.LOCK_UN)
      os.close(self._fd)
      self._fd = None

      # Try to remove the lock file, but don't try too hard because it is
      # unnecessary. This is mostly to help the user see whether a lock
      # exists by examining the filesystem.
      try:
         os.unlink(self._path)
      except:
         pass

'''  USAGE


with SimpleFlock("locktest", 2):  ## "locktest" is a temporary file that tells whether the lock is active
    ## perform action on the locked file(s)


## file is locked when with starts until its left
## if file is locked, code is paused until lock is released, then with is performed

'''



