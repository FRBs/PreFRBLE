#import FRB_analysis as FRB
#from FRB_analysis.Rays import *
import sys
from Rays import *

if len( sys.argv ) != 4:
    sys.exit( 'usgae: ipython MakeChoppedRays.py $N_tot $off $N_workers' )

span_tot = int(sys.argv[-3])
off = int(sys.argv[-2])
N_workers = int(sys.argv[-1])

span=span_tot/N_workers

r = [ range(i+off, i+span+off) for i in range(0,span_tot,span) ]

print r

from functools import partial
f = partial( CreateChoppedRaySegments, force=True )

pool = multiprocessing.Pool()
print 'get startded'
#pool.map( CreateChoppedRaySegments, r  )
pool.map( f, r  )
print 'all started'
pool.close()
pool.join()
print 'all joined'
