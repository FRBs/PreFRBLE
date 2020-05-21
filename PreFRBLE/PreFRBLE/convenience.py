from __future__ import print_function
import sys, h5py as h5, numpy as np, yt, csv
from time import time, sleep
from PreFRBLE.file_system import *
from PreFRBLE.parameter import *


from time import time
def TimeElapsed( func, *args, **kwargs ):
    """ measure time taken to compute function """
    def MeasureTime():
        t0 = time()
        res = func( *args, **kwargs)
        print( "{} took {} s".format( func.__name__, time()-t0 ) )
        return res
    return MeasureTime()
        



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
        #try:
            with h5.File( filename, 'a' ) as f:
                for data, key in zip( datas, keys ):
                    try:
                        f[key][()]
                        f.__delitem__( key )
                    except:
                        pass
                    f.create_dataset( key, data=data  )
            break
        #except:
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
    Returns -1 if no item satysfing the condition is found.

    >>> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first( (1,2,3), condition=lambda x: x > 9)
    -1

    
    THANKS TO Caridorc
    https://stackoverflow.com/questions/2361426/get-the-first-item-from-an-iterable-that-matches-a-condition
    
    """
    try:
        return next(x for x in iterable if condition(x))
    except:
        return -1




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
