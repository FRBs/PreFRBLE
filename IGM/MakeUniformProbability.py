'''
Obtain and compare P(RM) for the different models presented in Hackstein et al. 2018, available for download at https://crpropa.desy.de/ under "Additional Ressources"
these are given as data grid with uniform resolution, deduced from the final snapshot at z=0
the density distribution is the same for all the models, hence no investigation of DM or SM

'''


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathway import *
from Convenience import *
from Physics import *
import numpy as np



root_data = root+'MHD-models/clues/'

def MakeUniformProbability( model, N=1024, L=250e3/h, range=(1e-4,1e2), bins=100, log=True, posneg=False ):
    ## obtain probability of RM for a given model
    dx = L/N  ## resolution of uniform grid in kpc
    ## electron density in cm^-3
    rho = np.fromfile( root_data + 'mass-density_clues.dat', dtype='float32' ).reshape( N, N, N ) * critical_density*OmegaBaryon  / (proton_mass*1.16)  # transform gas density to electron number density
    ## magnetic field in muG
    B = np.fromfile( root_data + 'magnetic-field_clues_%s.dat' % model, dtype='float32' ).reshape( 3, N, N, N ) * 1e10  # Tesla -> muG
    if model == 'primordial':
        B /= 10.

    ## compute LoS integrals of RM, parallel to x and -z axis
    RM = []
    for i_ax, sgn in enumerate([-1,1]):
        RM.extend( np.sum( 0.81*1e3*dx*rho*B[i_ax] * sgn, axis=i_ax ) ) 
    RM = np.array(RM)
    if log and not posneg:
        RM = np.abs(RM)
        
    if posneg:
        ## compare distributions of opposite sign (makes sense with log=True)
        Pp, xp = histogram( RM, log=log, range=range, bins=bins, density=True )
        Pn, xn = histogram( -RM, log=log, range=range, bins=bins, density=True )
        P, x = [Pp, Pn], [xp, xn]
    else:
        P, x = histogram( RM, log=log, range=range, bins=bins, density=True )
        print Histogram2Expectation( P, x, log=log )
    return P, x
    


def PlotUniformProbability( models, N=1024, L=250e3/h, range=(1e-4,1e2), bins=100, log=True, density=True, posneg=False ):
    ## plot results for all models in a single panel
    fig, ax = plt.subplots()
    if log:
        plt.loglog()
    for model in models:
        P, x = MakeUniformProbability( model, N=N, L=L, range=range, bins=bins, posneg=posneg )
        if posneg:
            for x_, P_, l in zip(x,P, ['-','--']):
                plt.plot( x_[:-1]+np.diff(x_)/2, P_ * np.diff(x_)**(not density) * x_**(density), label=model if l=='-' else None, linestyle=l )
        else:
            x_ = x[:-1]+np.diff(x)/2
            plt.plot( x_, P * np.diff(x)**(not density) * x_**(density), label=model )

    ax.tick_params(labelsize=20)
    fig.subplots_adjust(bottom=0.2)
    plt.legend()
    plt.xlabel( r"|RM| / rad m$^{-2}$", fontdict={'size':16} )
    if density:
        plt.ylabel( 'P(|RM|)$\cdot$|RM|', fontdict={'size':16} )
    else:
        plt.ylabel( r"P(|RM|)$\Delta$|RM|", fontdict={'size':16} )
    plt.savefig( root_FRB + 'UniformProbability%s%s_RM.png' % ( 'posneg' if posneg else '', 'density' if density else '' ) )
    plt.close()

PlotUniformProbability( ['primordial', 'primordial2R', 'primordial3R', 'astrophysical', 'astrophysicalR', 'astrophysical1R' ], range=(2e-18,0.9e2), density=True) #, posneg=True )# , density=False )
