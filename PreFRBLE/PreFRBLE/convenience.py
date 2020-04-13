from __future__ import print_function
import sys, h5py as h5, numpy as np, yt, csv
from time import time, sleep
from PreFRBLE.file_system import *
from PreFRBLE.parameter import *



def CorrectScenario( measure='DM', **scenario ):
    """ this function is used to correct scenario keys wenn reading data, since some models have output stored under different name """
    
    result = scenario.copy()
    
    ## combine primordial and alpha results, former for DM, SM and tau, latter for RM (is identical to primordial, which was also computed)
    if 'RM' in measure:
        pass
        if 'IGM' in scenario:
            if 'primordial' in scenario['IGM'][0]:
                result['IGM'] = [scenario['IGM'][0].replace('primordial','alpha1-3rd')]
    else:
        if 'IGM' in scenario:
            if 'alpha' in scenario['IGM'][0]:
                result['IGM'] = [scenario['IGM'][0].replace(scenario['IGM'][0][:10], 'primordial' )]
    return result


## data keys inside likelihood files
def KeyLocal( model='Piro18/wind', measure='DM', axis='P' ):
    """ model key in likelihood_file_local """
    return '/'.join( [ model, measure, axis ] )

def KeyMilkyWay( model='JF12', measure='DM', axis='P', redshift=0.0  ):
    """ MW model key in likelihood_file_galaxy """
    return '/'.join( [ 'MilkyWay', model, measure, axis ] )

def KeyHost( redshift=0.0, model='Rodrigues18/smd', measure='DM', axis='P' ):
    """ host model key in likelihood_file_galaxy """
    return '/'.join( [ 'Host', model, '%.4f' % np.round( redshift, redshift_accuracy ), measure, axis ] )

def Keyinter( redshift=0.0, model='Rodrigues18', measure='DM', axis='P' ):
    """ intervening model key in likelihood_file_galaxy for case of galaxy intervening at redhsift """
    return '/'.join( [ 'inter', model, '%.4f' % np.round( redshift, redshift_accuracy ), measure, axis ] )

def KeyInter( redshift=0.0, model='Rodrigues18', measure='DM', axis='P' ):
    """ intervening model key in likelihood_file_galaxy for case of galaxy at unknown redshift along LoS to redshift """
    return '/'.join( [ 'Intervening', model, '%.4f' % np.round( redshift, redshift_accuracy ), measure, axis ] )


def KeyIGM( redshift=0.1, model='primordial', typ='far', nside=2**2, measure='DM', axis='P' ):  ## nside=2**6
    """ model key in likelihood_file_IGM """
#    print( measure, model, redshift )
    model_ = CorrectScenario( measure=measure, IGM=[model] )['IGM'][0]
#    print(model_)
    return '/'.join( [ model_, typ, str(nside), measure, '%.4f' % np.round( redshift, redshift_accuracy ), axis ] )

def KeyRedshift( population='flat', telescope='none', axis='P' ):
    """ model key in likelihood_file_redshift """
    return '/'.join( [ population, telescope, axis] )

#def KeyFull( measure='DM', axis='P', redshift=0.1, model_MW=['JF12'], model_IGM=['primordial'], model_Host=['Heesen11/IC10'], weight_Host='StarDensity_MW', model_Local=['Piro18/uniform_JF12'] ):
def KeyFull( measure='DM', axis='P', redshift=0.1, N_inter=False, **scenario ):
    """ scenario key in likelihood_file_Full """
    scenario_ = CorrectScenario( measure, **scenario )
    models = []
    for region in regions:
        model = scenario_.get( region )
        if model:
            models = np.append( models, model )
            if N_inter and region == 'Inter':
                models = np.append( models, "Ninter" )
    models = np.append( models, [ np.round( redshift, redshift_accuracy ), measure, axis ] )
    return '/'.join( models )


def KeyTelescope( measure='DM', axis='P', telescope='Parkes', population='SMD', **scenario ):
    """ scenario key in likelihood_file_telescope """
    scenario_ = CorrectScenario( measure, **scenario )
    models = [ telescope, population ]
    for region in regions:
        model = scenario_.get( region )
        if model:
            models = np.append( models, model )
    models = np.append( models, [ measure, axis ] )
    return '/'.join( models )


from time import sleep

## wrapper to write hdf5 files consistently
def Write2h5( filename='', datas=[], keys=[] ):
    """ conveniently write datas to keys in filename. overwrite existing entries """
    if type(keys) is str:
        sys.exit( 'Write2h5 needs list of datas and keys' )
    ### small workaround to allow for parallel computation. Use with caution, might corrupt nodes in your h5 file. in that case, visit:
    ### https://stackoverflow.com/questions/47979751/recover-data-from-corrupted-file/61147632?noredirect=1#comment108190378_61147632
    tries = 0
    while tries < 30:
        try:
            with h5.File( filename, 'a' ) as f:
                for data, key in zip( datas, keys ):
                    try:
                        f.__delitem__( key )
                    except:
                        pass
                    f.create_dataset( key, data=data  )
            break
        except:
            sleep(3e-2)
            tries += 1
            pass
    else:
        print(  "couldn't write ", keys )
        sys.exit(1)


## Read FRBcat

#FRB_dtype = [('ID','S'),('DM','f'),('DM_gal','f'), ('RM','f'),('tau','f'),('host_redshift','S'), ('tele','S')]
#FRB_dtype = [('ID','U9'),('DM','f'),('DM_gal','f'), ('RM','U10'),('tau','U10'),('host_redshift','U4'), ('tele','U10')]
FRB_dtype = [('ID','U9'),('DM','f'),('DM_gal','f'), ('RM','f'),('tau','f'),('host_redshift','f'), ('tele','U10')]

def GetFRBcat( telescopes=None, RM=None, tau=None, print_number=False ):
    """
    read all FRBs in FRBcat, downloaded to frbcat_file

    Parameters
    ----------
    telescopes : list
        list of considered telescopes, FRBs of other telescopes are ignored
    RM : boolean
        if True, only return FRBs observed with RM
    tau : boolean
        if True, only return FRBs observed with temproal broadening
    print_number : boolean
        if True, print number of extractet FRBs

    Returns
    -------
    FRBs : array
        structured numpy.array containing values listed in FRBcat

    """
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


def decode( string, dtype='U' ):
    """ short wrapper to decode byte-strings read from FRBcat """
    if 'f' in dtype:
        if 'null' in string:
            return float('NaN')
        return float(string)
    return string

def GetFRBsMeasures( measure='DM', FRBs=None ):
    """ returns measures of FRBs in FRBcat read with GetFRBcat() """
    if measure == 'DM':
        return FRBs['DM']-FRBs['DM_gal']
    elif measure == 'RM':
        return FRBs['RM']




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



def first(iterable, condition = lambda x: True):
    """
    Returns the first item in the `iterable` that satisfies the `condition`.
    If the condition is not given, returns the first item of the iterable.
    Raises `StopIteration` if no item satysfing the condition is found.

    >>> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first( () )
    Traceback (most recent call last):
    ...
    StopIteration
    
    THANKS TO Caridorc
    https://stackoverflow.com/questions/2361426/get-the-first-item-from-an-iterable-that-matches-a-condition
    
    """

    return next(x for x in iterable if condition(x))




## wrapper to show time needed for some function
'''
def HowLong( f, *args, print_additional='', **kwargs ):
    """ wrapper to print the time needed to call function f """
    t0 = time()
    ret = f( *args, **kwargs )
    t = time() - t0
    print( "Running %s took %i minutes and %.1f seconds %s" % (f.__name__, t//60, t%60, print_additional ) )
    return ret
'''
