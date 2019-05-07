import os
from glob import glob
from Models import *
from Parameters import *
from Rays import *
from multiprocessing import Pool
from time import time

t0 = time()

#files = glob( root_rays+model+'/*segment000*.h5' )
#files.sort()
#print files

#files = files[1022:1025]
#print files

'''
def CreateLoS( f ):
    ipix = int( f.split('pix')[-1].split('.h5')[0] )
    CreateLoSDMRM( ipix, remove=False, models=['primordial', 'astrophysical'] )
'''
CreateLoSsDMRM( models=['primordial', 'astrophysical'] )

'''   ## don't do in pool, makes h5 file corrupt due to parallel writing 
pool = Pool(8)
pool.map( CreateLoS, files )
'''
'''
for f in files:
    CreateLoS( f )
#    ipix = int( f.split('pix')[-1].split('.h5')[0] )
#    print ipix
#    CreateLoSDMRM( ipix )#, remove=False )
#'''


print 'took %.0f seconds' % ( time() - t0 )
