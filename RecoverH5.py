"""
This short script recovers all remaining data from .h5 file with broken nodes, (bad symbol table node signature)
"""



file_broken = 'likelihood/observables_likelihood_Full.h5'
file_recover = file_broken+".recover.h5"

import h5py as h5

def RecoverFile( f1, f2 ):
    """  recover read-open HDF5 file f1 to write-open HDF5 file f2  """
    names = []
    f1.visit(names.append)
    for n in names:
        try:
            f2.create_dataset( n, data=f1[n][()] )
        except:
            pass

with h5.File( file_broken, 'r' ) as fb:
    with h5.File( file_recover, 'w' ) as fr:
        for key in fb.keys():
            try:
                fr.create_group(key)
                RecoverFile( fb[key], fr[key] )
            except:
                fr.__delitem__(key)
        
        
