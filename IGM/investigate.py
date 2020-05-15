import matplotlib
matplotlib.use('Agg')
import h5py as h5, matplotlib.pyplot as plt
from glob import glob
from Physics import GasDensity, ScaleFactor
from Rays import FilenameSegment
from Parameters import *

root = '/work/stuf315/PreFRBLE/results/'

def PlotRay( ipix, expect=False ):
    ## find all files that belong to ray
    files = glob( FilenameSegment( ipix, -12345).replace('-12345','*') )
    files.sort()

    for f in files:
        ##   read and plot data (density, redshift)
        g = h5.File(f)['grid']
        z = g['redshift']
        i_snap = np.where( np.round(redshift_snapshots[1:], redshift_accuracy) >= z.value.max() )[0][0] ## !!! check if redshift_near is removed
        redshift_snapshot = z_snaps[ i_snap ]        

        a = ScaleFactor( z.value, redshift_snapshot )
        rho = g['Density']#*a**-3
        step=1
        plt.plot( z[::step], rho[::step], color='black' )

    if expect:
        ## plot expected GasDensity
        zs = np.arange( 0, 6, 0.05)
        Rho = GasDensity( zs )#*(1+zs)
        plt.plot( zs, Rho, linestyle='--', color='red' )


for i in range(1,16):
    PlotRay(i)
PlotRay(0, expect=True)
plt.yscale('log')
plt.savefig( root+'investigate rays.png' )
