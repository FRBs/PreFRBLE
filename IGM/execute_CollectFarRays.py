'''
Produce Probability functions of LoS observables for sources in the distant universe, outside the constrained volume
several snapshots of the simulation are combined using cosmological data stacking (e.g., da Silva et al. 2000)
Note that the light-travel distance between snapshots exceeds the simulation volume (Vazza-Hackstein models), hence LoS to high redshift are obtained by stacking segments, distributed randomly troughout the volume

'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Rays import CreateLoSsObservables


models=['primordial', 'astrophysical_mean', 'astrophysical_median', 'alpha1-3rd', 'alpha2-3rd', 'alpha3-3rd', 'alpha4-3rd', 'alpha5-3rd', 'alpha6-3rd', 'alpha7-3rd', 'alpha8-3rd', 'alpha9-3rd']  ## models to be considered for the magnetic field, provided as B~rho relations in relations_file

## first, clean folder and combine segments from previous runs to full LoS and reduce raw data to LoS observables, results saved to LoS_observables_file
#CreateLoSsObservables( models=models, remove=False, plot=True )
CreateLoSsObservables( models=models, remove=True )

'''
import numpy as np
from Physics import GasDensity

zs = np.arange( 0, 6, 0.05)
Rho = GasDensity( zs )#*(1+zs)
plt.plot( zs, Rho, linestyle='--', color='red' )
plt.yscale('log')
root = '/work/stuf315/PreFRBLE/results/'
plt.savefig( root+'investigate rays' )
plt.close()

from Plots import PlotFarRays
#PlotFarRays( measure='SM', plot_mean=True, uniform=True )
'''
